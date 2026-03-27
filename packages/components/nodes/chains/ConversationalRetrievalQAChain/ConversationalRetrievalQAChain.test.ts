import { BaseCallbackHandler } from '@langchain/core/callbacks/base'
import { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import { Document } from '@langchain/core/documents'
import { BaseMessage } from '@langchain/core/messages'
import { SimpleChatModel } from '@langchain/core/language_models/chat_models'
import { BaseRetriever } from '@langchain/core/retrievers'

const mockCreatedHandlers: MockCustomChainHandler[] = []

class MockCustomChainHandler extends BaseCallbackHandler {
    name = 'mock_custom_chain_handler'
    sseStreamer: any
    chatId: string
    skipK: number
    initialSkipK: number
    returnSourceDocuments: boolean
    isLLMStarted = false
    currentInvocationSkipped = false

    constructor(sseStreamer: any, chatId: string, skipK?: number, returnSourceDocuments?: boolean) {
        super()
        this.sseStreamer = sseStreamer
        this.chatId = chatId
        this.skipK = skipK ?? 0
        this.initialSkipK = this.skipK
        this.returnSourceDocuments = returnSourceDocuments ?? false
        mockCreatedHandlers.push(this)
    }

    copy() {
        return this
    }

    handleLLMStart() {
        this.isLLMStarted = false
    }

    handleLLMNewToken(token: string) {
        if (token.startsWith('__CALL_START__:')) {
            this.isLLMStarted = false
            if (this.skipK > 0) {
                this.skipK -= 1
                this.currentInvocationSkipped = true
                return
            }
            this.currentInvocationSkipped = false
            return
        }

        if (this.currentInvocationSkipped) return
        if (!this.isLLMStarted) {
            this.isLLMStarted = true
            this.sseStreamer?.streamStartEvent(this.chatId, token)
        }
        this.sseStreamer?.streamTokenEvent(this.chatId, token)
    }

    handleLLMEnd() {
        if (this.currentInvocationSkipped) return
        this.sseStreamer?.streamEndEvent(this.chatId)
    }

    handleChainEnd(outputs: Record<string, any>) {
        if (this.returnSourceDocuments) {
            this.sseStreamer?.streamSourceDocumentsEvent(this.chatId, outputs?.sourceDocuments)
        }
    }
}

jest.mock('../../../src/handler', () => {
    class MockConsoleCallbackHandler extends BaseCallbackHandler {
        name = 'mock_console_callback_handler'

        copy() {
            return this
        }
    }

    return {
        ConsoleCallbackHandler: MockConsoleCallbackHandler,
        CustomChainHandler: MockCustomChainHandler,
        additionalCallbacks: jest.fn().mockResolvedValue([])
    }
})

class MockStreamingChatModel extends SimpleChatModel {
    _llmType() {
        return 'mock-streaming-chat-model'
    }

    async _call(messages: BaseMessage[], _options: this['ParsedCallOptions'], runManager?: CallbackManagerForLLMRun): Promise<string> {
        const content = messages
            .map((message) => (typeof message.content === 'string' ? message.content : JSON.stringify(message.content)))
            .join('\n')

        const invocationType = content.includes('CONDENSE') ? 'condense' : 'response'
        const tokens = invocationType === 'condense' ? ['standalone-question'] : ['final', ' answer']

        await runManager?.handleLLMNewToken(`__CALL_START__:${invocationType}`)

        for (const token of tokens) {
            await runManager?.handleLLMNewToken(token)
        }

        return tokens.join('')
    }
}

class MockRetriever extends BaseRetriever {
    docs: Document[]
    lc_namespace = ['tests', 'mockRetriever']

    constructor(docs: Document[]) {
        super()
        this.docs = docs
    }

    async _getRelevantDocuments(): Promise<Document[]> {
        return this.docs
    }
}

const { nodeClass: ConversationalRetrievalQAChainNode } = require('./ConversationalRetrievalQAChain')

describe('ConversationalRetrievalQAChain streaming', () => {
    const sessionId = 'session-1'
    const chatId = 'chat-1'
    const sourceDocuments = [new Document({ pageContent: 'retrieved-doc', metadata: { id: 'doc-1' } })]

    beforeEach(() => {
        jest.clearAllMocks()
        mockCreatedHandlers.length = 0
    })

    const createNodeData = (memory: any) => ({
        inputs: {
            model: new MockStreamingChatModel({}),
            vectorStoreRetriever: new MockRetriever(sourceDocuments),
            memory,
            returnSourceDocuments: true,
            rephrasePrompt: 'CONDENSE {chat_history} :: {question}',
            responsePrompt: 'Answer from context: {context}'
        }
    })

    const createOptions = (sseStreamer: any, shouldStreamResponse = true) => ({
        shouldStreamResponse,
        sseStreamer,
        chatId,
        logger: { verbose: jest.fn() },
        orgId: 'org-1'
    })

    it('returns the final answer, source documents, and configures streaming without history', async () => {
        const memory = {
            getChatMessages: jest.fn().mockResolvedValue([]),
            addChatMessages: jest.fn().mockResolvedValue(undefined)
        }
        const sseStreamer = {
            streamStartEvent: jest.fn(),
            streamTokenEvent: jest.fn(),
            streamEndEvent: jest.fn(),
            streamSourceDocumentsEvent: jest.fn()
        }

        const node = new ConversationalRetrievalQAChainNode({ sessionId })
        const result = await node.run(createNodeData(memory), 'What is the answer?', createOptions(sseStreamer))

        expect(result).toEqual({ text: 'final answer', sourceDocuments })
        expect(mockCreatedHandlers).toHaveLength(1)
        expect(mockCreatedHandlers[0].initialSkipK).toBe(0)
        expect(mockCreatedHandlers[0].skipK).toBe(0)
        expect(memory.addChatMessages).toHaveBeenCalledWith(
            [
                { text: 'What is the answer?', type: 'userMessage' },
                { text: 'final answer', type: 'apiMessage' }
            ],
            sessionId
        )
    })

    it('configures the streaming handler to skip the condense-question call when history exists', async () => {
        const memory = {
            getChatMessages: jest.fn().mockResolvedValue([
                { message: 'Previous question', type: 'userMessage' },
                { message: 'Previous answer', type: 'apiMessage' }
            ]),
            addChatMessages: jest.fn().mockResolvedValue(undefined)
        }
        const sseStreamer = {
            streamStartEvent: jest.fn(),
            streamTokenEvent: jest.fn(),
            streamEndEvent: jest.fn(),
            streamSourceDocumentsEvent: jest.fn()
        }

        const node = new ConversationalRetrievalQAChainNode({ sessionId })
        const result = await node.run(createNodeData(memory), 'Follow-up question?', createOptions(sseStreamer))

        expect(result).toEqual({ text: 'final answer', sourceDocuments })
        expect(mockCreatedHandlers).toHaveLength(1)
        expect(mockCreatedHandlers[0].initialSkipK).toBe(1)
        expect(mockCreatedHandlers[0].skipK).toBe(0)
        expect(memory.addChatMessages).toHaveBeenCalledWith(
            [
                { text: 'Follow-up question?', type: 'userMessage' },
                { text: 'final answer', type: 'apiMessage' }
            ],
            sessionId
        )
    })

    it('does not create a streaming handler when streaming is disabled', async () => {
        const memory = {
            getChatMessages: jest.fn().mockResolvedValue([]),
            addChatMessages: jest.fn().mockResolvedValue(undefined)
        }
        const sseStreamer = {
            streamStartEvent: jest.fn(),
            streamTokenEvent: jest.fn(),
            streamEndEvent: jest.fn(),
            streamSourceDocumentsEvent: jest.fn()
        }

        const node = new ConversationalRetrievalQAChainNode({ sessionId })
        const result = await node.run(createNodeData(memory), 'No streaming please', createOptions(sseStreamer, false))

        expect(result).toEqual({ text: 'final answer', sourceDocuments })
        expect(mockCreatedHandlers).toHaveLength(0)
        expect(memory.addChatMessages).toHaveBeenCalledWith(
            [
                { text: 'No streaming please', type: 'userMessage' },
                { text: 'final answer', type: 'apiMessage' }
            ],
            sessionId
        )
    })
})
