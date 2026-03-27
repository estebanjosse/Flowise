import path from 'path'
import { Document } from '@langchain/core/documents'
import { RunnableLambda } from '@langchain/core/runnables'
import { ICommonObject, IMessage, INodeData, IServerSideEventStreamer } from '../../../src/Interface'

jest.mock('../../../src/handler', () => ({
    ConsoleCallbackHandler: jest.fn().mockImplementation(() => ({})),
    additionalCallbacks: jest.fn().mockResolvedValue([])
}))

const { nodeClass: ConversationalRetrievalQAChainNode } = require('./ConversationalRetrievalQAChain')
const langChainCorePackagePath = require.resolve('@langchain/core/package.json')
const { FakeListChatModel } = require(path.join(path.dirname(langChainCorePackagePath), 'dist/utils/testing/chat_models.cjs'))

type HarnessOptions = {
    history?: IMessage[]
    responses?: string[]
    returnSourceDocuments?: boolean
    shouldStreamResponse?: boolean
}

const createSseStreamer = () =>
    ({
        streamStartEvent: jest.fn(),
        streamTokenEvent: jest.fn(),
        streamSourceDocumentsEvent: jest.fn(),
        streamEndEvent: jest.fn()
    } as unknown as jest.Mocked<IServerSideEventStreamer>)

const createMemory = (history: IMessage[] = []) => ({
    getChatMessages: jest.fn().mockResolvedValue(history),
    addChatMessages: jest.fn().mockResolvedValue(undefined),
    clearChatMessages: jest.fn().mockResolvedValue(undefined)
})

const createHarness = ({
    history = [],
    responses = ['Answer from retrieved docs'],
    returnSourceDocuments = true,
    shouldStreamResponse = true
}: HarnessOptions = {}) => {
    const model = new FakeListChatModel({ responses })
    const retriever = RunnableLambda.from(async () => [
        new Document({ pageContent: 'Relevant context from the retriever', metadata: { id: 'doc-1' } })
    ])
    const memory = createMemory(history)
    const sseStreamer = createSseStreamer()
    const node = new ConversationalRetrievalQAChainNode({ sessionId: 'test-session' })

    const nodeData = {
        id: 'node-1',
        label: 'Conversational Retrieval QA Chain',
        name: 'conversationalRetrievalQAChain',
        type: 'ConversationalRetrievalQAChain',
        icon: 'qa.svg',
        version: 3,
        category: 'Chains',
        baseClasses: ['ConversationalRetrievalQAChain'],
        inputs: {
            model,
            vectorStoreRetriever: retriever,
            memory,
            returnSourceDocuments
        }
    } as unknown as INodeData

    const options: ICommonObject = {
        shouldStreamResponse,
        sseStreamer,
        chatId: 'chat-1'
    }

    return { node, nodeData, options, memory, model, retriever, sseStreamer }
}

describe('ConversationalRetrievalQAChain test harness', () => {
    it('runs the node with focused streaming callback test doubles', async () => {
        const harness = createHarness()

        const result = await harness.node.run(harness.nodeData, 'What does the document say?', harness.options)

        expect(result).toEqual({
            text: 'Answer from retrieved docs',
            sourceDocuments: [
                expect.objectContaining({
                    pageContent: 'Relevant context from the retriever',
                    metadata: { id: 'doc-1' }
                })
            ]
        })
        expect(harness.memory.getChatMessages).toHaveBeenCalledWith('test-session', false, undefined)
        expect(harness.memory.addChatMessages).toHaveBeenCalledWith(
            [
                {
                    text: 'What does the document say?',
                    type: 'userMessage'
                },
                {
                    text: 'Answer from retrieved docs',
                    type: 'apiMessage'
                }
            ],
            'test-session'
        )
        expect(harness.sseStreamer.streamEndEvent).toHaveBeenCalledWith('chat-1')
    })
})
