import asyncio
import json
import random
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

logger = structlog.get_logger()

PRODUCT_TEMPLATE = """{{#system~}}
You a creative product designer.
{{~/system}}
{{#user~}}
Your task is to generate lists of commercial products.
{{product_description}}
List the products one per line. Don't number them or print any other text, just print a product on each line.

Here is an example:
{{#each examples~}}
{{this}}
{{/each}}
Now generate a list of {{n_products}} products.
{{~/user}}
{{#assistant~}}
{{gen 'list' temperature=0.7 max_tokens=1000}}
{{~/assistant}}"""

DYSTOPIAN_EXAMPLES = [
    "A pill that makes you forget your ex",
    "A service that lets you rent a friend",
    "A brain implant to control your emotions",
    "A billboard that can read the minds of passersby to personalize ads",
]
DYSTOPIAN_DESCRIPTION = "The products should be useful, but also dystopian."

SUPERLATIVE_EXAMPLES = [
    "The thinnest phone on the Planet",
    "The most comfortable shoes ever",
    "The friendliest accountant in the country",
    "The most caffeinated soda in the world",
]
SUPERLATIVE_DESCRIPTION = (
    "The products should be superlatives. This defines their unique selling point."
)

ABSURD_EXAMPLES = [
    "A scarf for your fingers",
    "A service for waking you up in the morning with confetti",
    "A gym where you can only work out in the dark",
    "A dating app for pets",
    "Wall insulation made out of snakes",
    "An insurance for your painted nails",
]
ABSURD_DESCRIPTION = (
    "The products should be absurd. They should be funny and not make sense."
)

AAS_EXAMPLES = [
    "Ice cream as a service",
    "Hugs as a service",
    "Ambiance as a service",
]
AAS_DESCRIPTION = "The products should be 'as a service' products. They should be plausible yet farfetched."


class ProductGenerator:
    def __init__(  # noqa: PLR0913
        self,
        llm: LLM,
        n_products: int,
        n_iter: int,
        examples: list[str],
        product_description: str,
    ):
        self._llm = llm
        self.n_products = n_products
        self.n_iter = n_iter
        self.examples = examples
        self.product_description = product_description
        logger.info("Initialized products", n_products=n_products, n_iter=n_iter)

    async def awrite_product(self, program: Program, **kwargs) -> Program:
        """For some reason the program await is messed up so we have to wrap in this async function"""
        return await program(**kwargs)

    async def awrite(self) -> list[str]:
        logger.info("Async writing products")

        tasks = []
        for _ in range(self.n_iter):
            product = Program(text=PRODUCT_TEMPLATE, llm=self._llm, async_mode=True)
            tasks.append(
                asyncio.create_task(
                    self.awrite_product(
                        product,
                        examples=self.examples,
                        n_products=self.n_products,
                        product_description=self.product_description,
                    )
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        products = self.collect(results)
        products += self.examples
        unique_products = list(set(products))
        logger.info(
            f"Generated {len(unique_products)} unique products",
            products=unique_products,
        )
        return unique_products

    def collect(self, results: list[Program]) -> list[str]:
        all_products = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Error generating products", error=r)
            else:
                logger.info(r)
                products = r["list"].split("\n")
                products = [t.strip().replace("-", " ") for t in products]
                all_products.extend(products)
        return all_products


class Products:
    def __init__(
        self,
        llm: LLM,
        n_products: int,
        n_iter: int,
        tmp_dir: Path = Path("tmp/"),
    ):
        self._llm = llm
        self.n_products = n_products
        self.n_iter = n_iter
        self._generators = self.init_generators()
        self._output_dir = tmp_dir
        logger.info("Initialized products", n_products=n_products, n_iter=n_iter)

    async def awrite(self) -> bool:
        logger.info("Async writing products")
        tasks = []
        for g in self._generators:
            tasks.append(asyncio.create_task(g.awrite()))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        products = flatten(results)
        self.save(products)
        return True

    def init_generators(self) -> list[ProductGenerator]:
        superlative = ProductGenerator(
            llm=self._llm,
            n_products=self.n_products,
            n_iter=self.n_iter,
            examples=SUPERLATIVE_EXAMPLES,
            product_description=SUPERLATIVE_DESCRIPTION,
        )
        dystopian = ProductGenerator(
            llm=self._llm,
            n_products=self.n_products,
            n_iter=self.n_iter,
            examples=DYSTOPIAN_EXAMPLES,
            product_description=DYSTOPIAN_DESCRIPTION,
        )
        absurd = ProductGenerator(
            llm=self._llm,
            n_products=self.n_products,
            n_iter=self.n_iter,
            examples=ABSURD_EXAMPLES,
            product_description=ABSURD_DESCRIPTION,
        )
        as_a_service = ProductGenerator(
            llm=self._llm,
            n_products=self.n_products,
            n_iter=self.n_iter,
            examples=AAS_EXAMPLES,
            product_description=AAS_DESCRIPTION,
        )
        return [superlative, dystopian, absurd, as_a_service]

    def save(self, products: list[str]) -> None:
        with (self._output_dir / "advert-products.json").open("w") as f:
            json.dump(products, f, indent=2)


def flatten(things: list) -> list:
    return [e for nested_things in things for e in nested_things]


PRODUCTS_PATH = Path(__file__).parent / "resources" / "advert-products.json"
PRODUCTS_CACHE = None


def load_products() -> list[str]:
    global PRODUCTS_CACHE  # noqa: PLW0603
    if PRODUCTS_CACHE is None:
        with PRODUCTS_PATH.open() as f:
            PRODUCTS_CACHE = json.load(f)
    return PRODUCTS_CACHE


def random_product() -> str:
    """Return a random product from the list of products"""
    return random.choice(load_products())
