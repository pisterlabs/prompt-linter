import cohere
import numpy as np
import pickle

def layer_1_embeddings(user_input):
    with open('cohere_apikey.txt') as f:
        cohere_api_key = f.readline()
    co = cohere.Client(cohere_api_key)
    
    # Replace once we have the pickles

    # texts = pickle.load(open("readme.pkl"))
    texts = [
        """An enterprise-class UI design language and React UI library.

CI status codecov NPM version NPM downloads

Renovate status Total alerts  

Follow Twitter FOSSA Status Discussions  Issues need help

English | Português | 简体中文 | Українською | Spanish | 日本語 FeaturesEnterprise-class UI designed for web applications.A set of high-quality React components out of the box.Written in TypeScript with predictable static types.Whole package of design resources and development tools.Internationalization support for dozens of languages.Powerful theme customization in every detail.""",
        """Create React apps with no build configuration.

Creating an App – How to create a new app.
User Guide – How to develop apps bootstrapped with Create React App.
Create React App works on macOS, Windows, and Linux.
If something doesn’t work, please file an issue.
If you have questions or need help, please ask in GitHub Discussions.

Quick Overview
npx create-react-app my-app
cd my-app
npm start
If you've previously installed create-react-app globally via npm install -g create-react-app, we recommend you uninstall the package using npm uninstall -g create-react-app or yarn global remove create-react-app to ensure that npx always uses the latest version.""",
        """Dotenv is a zero-dependency module that loads environment variables from a .env file into process.env. Storing configuration in the environment separate from code is based on The Twelve-Factor App methodology.

BuildStatus Build status NPM version js-standard-style Coverage Status LICENSE Conventional Commits Featured on Openbase Limited Edition Tee Original Limited Edition Tee Redacted

Install
# install locally (recommended)
npm install dotenv --save
Or installing with yarn? yarn add dotenv""",
        """Fast, unopinionated, minimalist web framework for Node.js.

NPM Version NPM Install Size NPM Downloads

const express = require('express')
const app = express()

app.get('/', function (req, res) {
  res.send('Hello World')
})

app.listen(3000)
Installation
This is a Node.js module available through the npm registry.

Before installing, download and install Node.js. Node.js 0.10 or higher is required.

If this is a brand new project, make sure to create a package.json first with the npm init command.

Installation is done using the npm install command:""",
    """An implementation of JSON Web Tokens.

This was developed against draft-ietf-oauth-json-web-token-08. It makes use of node-jws

Install
$ npm install jsonwebtoken
Migration notes
From v7 to v8
Usage
jwt.sign(payload, secretOrPrivateKey, [options, callback])
(Asynchronous) If a callback is supplied, the callback is called with the err or the JWT.

(Synchronous) Returns the JsonWebToken as string

payload could be an object literal, buffer or string representing valid JSON."""
    ]
    num_texts = len(texts)
    pkgs, _ = pickle.load(open('pkg.pkl', 'rb'))
    # print(pkgs[:5])
    pkgs = pkgs[:num_texts]

    response = co.embed(
        model='large',
        texts=texts)

    prompt = co.embed(model='large', texts=[user_input]).embeddings[0]

    def cos_sim(v1, v2):
        return np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)

    similarities = [(cos_sim(response.embeddings[i], prompt), pkgs[i]) for i in range(num_texts)]
    similarities.sort(reverse=True)

    return [similarities[i][1] for i in range(min(len(texts), 20))]

if __name__ == "__main__":
    print(layer_1_embeddings("I want to handle jsons"))
