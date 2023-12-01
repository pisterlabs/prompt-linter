# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import langchain_pb2 as langchain__pb2


class LangChainStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CleanAdvisory = channel.unary_unary(
                '/langchain.LangChain/CleanAdvisory',
                request_serializer=langchain__pb2.CleanAdvisoryRequest.SerializeToString,
                response_deserializer=langchain__pb2.CleanAdvisoryResponse.FromString,
                )
        self.Chat = channel.unary_unary(
                '/langchain.LangChain/Chat',
                request_serializer=langchain__pb2.ChatRequest.SerializeToString,
                response_deserializer=langchain__pb2.ChatResponse.FromString,
                )
        self.CleanSnippets = channel.unary_unary(
                '/langchain.LangChain/CleanSnippets',
                request_serializer=langchain__pb2.CleanSnippetsRequest.SerializeToString,
                response_deserializer=langchain__pb2.CleanSnippetsResponse.FromString,
                )


class LangChainServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CleanAdvisory(self, request, context):
        """rpc Summarize(SummarizeRequest) returns (SummarizeResponse);
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Chat(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CleanSnippets(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LangChainServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CleanAdvisory': grpc.unary_unary_rpc_method_handler(
                    servicer.CleanAdvisory,
                    request_deserializer=langchain__pb2.CleanAdvisoryRequest.FromString,
                    response_serializer=langchain__pb2.CleanAdvisoryResponse.SerializeToString,
            ),
            'Chat': grpc.unary_unary_rpc_method_handler(
                    servicer.Chat,
                    request_deserializer=langchain__pb2.ChatRequest.FromString,
                    response_serializer=langchain__pb2.ChatResponse.SerializeToString,
            ),
            'CleanSnippets': grpc.unary_unary_rpc_method_handler(
                    servicer.CleanSnippets,
                    request_deserializer=langchain__pb2.CleanSnippetsRequest.FromString,
                    response_serializer=langchain__pb2.CleanSnippetsResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'langchain.LangChain', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LangChain(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CleanAdvisory(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/langchain.LangChain/CleanAdvisory',
            langchain__pb2.CleanAdvisoryRequest.SerializeToString,
            langchain__pb2.CleanAdvisoryResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Chat(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/langchain.LangChain/Chat',
            langchain__pb2.ChatRequest.SerializeToString,
            langchain__pb2.ChatResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CleanSnippets(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/langchain.LangChain/CleanSnippets',
            langchain__pb2.CleanSnippetsRequest.SerializeToString,
            langchain__pb2.CleanSnippetsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
