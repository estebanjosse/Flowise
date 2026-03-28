import path from 'path'
import { AIMessage } from '@langchain/core/messages'
import { Document } from '@langchain/core/documents'
import { RunnableLambda } from '@langchain/core/runnables'
import { ICommonObject, IMessage, INodeData, IServerSideEventStreamer } from '../../../src/Interface'

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

const createStreamingCapableModel = (responses: string[], shouldStreamResponse: boolean) => {
    const model = new FakeListChatModel({ responses })

    if (shouldStreamResponse) {
        let responseIndex = 0
        model._generate = async (
            _messages: unknown,
            _options: unknown,
            runManager: { handleLLMNewToken?: (token: string) => Promise<void> }
        ) => {
            const response = responses[Math.min(responseIndex, responses.length - 1)] ?? ''
            responseIndex += 1

            for (const token of response) {
                await runManager?.handleLLMNewToken?.(token)
            }

            return {
                generations: [
                    {
                        text: response,
                        message: new AIMessage(response)
                    }
                ],
                llmOutput: {}
            }
        }
    }

    return model
}

const createHarness = ({
    history = [],
    responses = ['Answer from retrieved docs'],
    returnSourceDocuments = true,
    shouldStreamResponse = true
}: HarnessOptions = {}) => {
    const model = createStreamingCapableModel(responses, shouldStreamResponse)
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
        chatId: 'chat-1',
        logger: {
            verbose: jest.fn(),
            debug: jest.fn(),
            info: jest.fn(),
            warn: jest.fn(),
            error: jest.fn()
        }
    }

    return { node, nodeData, options, memory, model, retriever, sseStreamer }
}

describe('ConversationalRetrievalQAChain test harness', () => {
    it('runs the node with focused streaming callback test doubles', async () => {
        const harness = createHarness()

        const result = await harness.node.run(harness.nodeData, 'What does the document say?', harness.options)
        await new Promise((resolve) => setTimeout(resolve, 0))

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

    it('streams progressive answer tokens during a streaming run', async () => {
        const harness = createHarness({
            responses: ['Progressive streaming answer']
        })

        await harness.node.run(harness.nodeData, 'Stream the answer progressively', harness.options)
        await new Promise((resolve) => setTimeout(resolve, 0))

        const streamedTokens = harness.sseStreamer.streamTokenEvent.mock.calls.map(([, token]) => token)

        expect(harness.sseStreamer.streamStartEvent).toHaveBeenCalled()
        expect(streamedTokens.length).toBeGreaterThan(1)
        expect(streamedTokens.join('')).toBe('Progressive streaming answer')
    })

    it('emits source documents and does not leak condensed-question text in streamed tokens', async () => {
        const condensedQuestionText = 'CONDENSED QUESTION SHOULD NOT STREAM'
        const finalAnswerText = 'Final streamed answer from retrieved docs'
        const harness = createHarness({
            history: [
                { message: 'Earlier user question', type: 'userMessage' },
                { message: 'Earlier assistant answer', type: 'apiMessage' }
            ],
            responses: [condensedQuestionText, finalAnswerText],
            returnSourceDocuments: true,
            shouldStreamResponse: true
        })

        await harness.node.run(harness.nodeData, 'Follow-up question', harness.options)
        await new Promise((resolve) => setTimeout(resolve, 0))

        const streamedTokens = harness.sseStreamer.streamTokenEvent.mock.calls.map(([, token]) => token)
        const streamedText = streamedTokens.join('')

        expect(streamedText).toBe(finalAnswerText)
        expect(streamedText).not.toContain(condensedQuestionText)
        expect(harness.sseStreamer.streamSourceDocumentsEvent).toHaveBeenCalledWith('chat-1', [
            expect.objectContaining({
                pageContent: 'Relevant context from the retriever',
                metadata: { id: 'doc-1' }
            })
        ])
    })
})
