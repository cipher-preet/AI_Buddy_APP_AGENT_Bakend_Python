import os

import pytest

from services.search.parallel import ParallelSearchProvider
from services.search.provider import SearchRequest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("PARALLEL_API_KEY"), reason="PARALLEL_API_KEY not set")
@pytest.mark.asyncio
async def test_live_parallel_search_smoke():
    provider = ParallelSearchProvider()
    result = await provider.search(
        SearchRequest(
            objective="Find typical qualifications for a software engineering role.",
            search_queries=["software engineer qualifications", "software engineer requirements"],
            mode="fast",
            max_results=3,
        )
    )
    assert not result.degraded or result.results
    if result.results:
        assert result.results[0].url.startswith("http")
