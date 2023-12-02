from git import Repo
from langchain.document_loaders import GitLoader


def api3_github():
    return [
        "https://github.com/api3dao/vitepress-docs",
        "https://github.com/api3dao/airnode",
        "https://github.com/api3dao/signed-api",
        "https://github.com/api3dao/airseeker-v2",
        "https://github.com/api3dao/wallet-watcher",
    ]


def alchemy_github():
    return [
        # Alchemy
        "https://github.com/alchemyplatform/alchemy-sdk-js",
        "https://github.com/alchemyplatform/light-account",
        "https://github.com/alchemyplatform/ERC-6900-Ref-Implementation",
    ]


def apeworx_github():
    return [
        "https://github.com/ApeWorX/ape",
        "https://github.com/ApeWorX/silverback",
        "https://github.com/ApeWorX/py-tokenlists",
    ]


def arbitrum_github():
    return [
        "https://github.com/ArbitrumFoundation/docs",
        "https://github.com/ArbitrumFoundation/governance",
        "https://github.com/ArbitrumFoundation/sybil-detection",
        "https://github.com/ArbitrumFoundation/docs",
        "https://github.com/ArbitrumFoundation/docs",
    ]


def chainlink_github():
    return [
        # Chainlink
        "https://github.com/smartcontractkit/chainlink",
        "https://github.com/smartcontractkit/ccip",
        "https://github.com/smartcontractkit/external-adapters-js",
        "https://github.com/smartcontractkit/LinkToken",
    ]


def foundry_github():
    return [
        # Foundry
        "https://github.com/foundry-rs/book",
        "https://github.com/foundry-rs/forge-std",
        "https://github.com/foundry-rs/forge-template",
    ]


def metamask_github():
    return [
        # MetaMask
        "https://github.com/MetaMask/metamask-extension",
        "https://github.com/MetaMask/metamask-mobile",
        "https://github.com/MetaMask/metamask-docs",
        "https://github.com/MetaMask/metamask-sdk",
        "https://github.com/MetaMask/snaps",
        "https://github.com/MetaMask/core",
    ]


def openzeppelin_github():
    return [
        # Open Zeppelin
        "https://github.com/OpenZeppelin/openzeppelin-contracts",
        "https://github.com/OpenZeppelin/ethernaut",
        "https://github.com/OpenZeppelin/defender-client",
        "https://github.com/OpenZeppelin/openzeppelin-contracts-upgradeable",
        "https://github.com/OpenZeppelin/solidity-docgen",
    ]


def slither_github():
    return [
        # Smart Contract Auditing
        "https://github.com/crytic/slither",
        "https://github.com/crytic/echidna",
        "https://github.com/crytic/building-secure-contracts",
        "https://github.com/crytic/solc-select",
        "https://github.com/crytic/properties",
    ]
