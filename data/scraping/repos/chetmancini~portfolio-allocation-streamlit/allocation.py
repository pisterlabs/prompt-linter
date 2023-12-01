from typing import Dict
import streamlit as st
from portfolio_app.portfolio.models import SecurityAllocation
from portfolio_app.provider.openai import OpenAIClient


class AllocationCache:
    """
    Facade of multiple caches to guarantee access to repeated data
    """

    def __init__(self):
        self.cache: Dict[str, SecurityAllocation] = {}
        st.session_state["allocation_cache"] = {}

    def get(self, symbol: str) -> SecurityAllocation:
        if symbol in self.cache:
            return self.cache[symbol]
        session_cache = st.session_state.get("allocation_cache", {})
        if symbol in session_cache:
            return session_cache[symbol]
        return None

    def exists(self, symbol: str) -> bool:
        if symbol in self.cache:
            return True
        session_cache = st.session_state.get("allocation_cache", {})
        if symbol in session_cache:
            return True
        return False

    def set(self, symbol: str, allocation: SecurityAllocation) -> None:
        self.cache[symbol] = allocation
        session_cache = st.session_state.get("allocation_cache", {})
        session_cache[symbol] = allocation
        st.session_state["allocation_cache"] = session_cache


class AllocationLookupService:
    def __init__(self):
        self.cache = AllocationCache()
        api_key = st.session_state.get("openai_api_key")
        if api_key:
            print("Using OpenAI API Key from session state")
        self.openai_client = OpenAIClient(api_key=api_key)

    def get_allocations_by_symbol(self, symbol: str) -> SecurityAllocation:
        if self.cache.exists(symbol):
            return self.cache.get(symbol)

        response = self.openai_client.lookup_allocation(symbol=symbol)
        self.cache.set(symbol=symbol, allocation=response)
        return response

    async def get_allocations_by_symbol_async(self, symbol: str) -> SecurityAllocation:
        if self.cache.exists(symbol):
            return self.cache.get(symbol)

        response = await self.openai_client.lookup_allocation_async(symbol=symbol)
        self.cache.set(symbol=symbol, allocation=response)
        return response
