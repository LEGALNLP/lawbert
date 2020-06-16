import asyncio
import logging
import sys
from typing import IO
import urllib.error
import urllib.parse
import os

import aiofiles
import aiohttp
from aiohttp import ClientSession

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.DEBUG,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
DOWNLOAD_PATH = os.path.join("data", "pdfs")

logger = logging.getLogger(__name__)


async def fetch_html(url: str, session: ClientSession, **kwargs) -> None:
    """ Get request async wrapper to fetch PDF files

    Args:
        url (str): [description]
        session (ClientSession): [description]
    """
    try:
        resp = await session.request(url=url, method="GET", **kwargs)
        resp.raise_for_status()
        logger.info(f"Got response for {url} -> {resp.status}")
    except Exception as e:
        logger.error(f"{url} -> {str(e)}")
    else:
        f_name = url.split("/")[-1]
        logger.info(f_name)
        response = await resp.read()
        with open(os.path.join(DOWNLOAD_PATH, f_name), "wb") as f:
            f.write(response)
            logger.info(
                f"Write successful {url} -> {os.path.join(DOWNLOAD_PATH, f_name)}"
            )


async def bulk_crawl_and_write(urls: set, **kwargs) -> None:
    """Crawl concurrently and write to a new file
    """
    async with ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(fetch_html(url=url, session=session, **kwargs))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    with open(os.path.join("data", "links.txt")) as f:
        urls = set(map(str.strip, f))

    asyncio.run(bulk_crawl_and_write(urls=urls))
