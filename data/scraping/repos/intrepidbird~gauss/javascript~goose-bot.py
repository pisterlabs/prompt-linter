import { config } from "dotenv";

config();

import OpenAI from 'openai';

const openai = new OpenAI({
    apiKey: 'placeholder', // defaults to process.env["OPENAI_API_KEY"]
    baseURL: "http://100.127.26.132:7869"
});

import { Client } from "@projectdysnomia/dysnomia";

import fetch from "node-fetch";

const bot = new Client(process.env.TOKEN, {
    gateway: {
        intents: [
            "guildMessages",
            "directMessages",
            "messageContent"
        ]
    }
});

bot.on("ready", () => {
    console.log("Ready!");
});

bot.on("error", (err) => {
    console.error(err);
});

bot.on("messageCreate", async (msg) => {
    const message = msg;
    if (msg.content) {
        let content = msg.content;
        if (content.startsWith("goose ")) {
            // let channel = await bot.getChannel(message.channel.id);
            // console.log("Got channel", channel);
            let outputMessage = await bot.createMessage(message.channel.id, {
                content: "Downloading context...",
                messageReference: {
                    messageID: message.id
                }
            });
            let instruction = content.slice("goose ".length);
            let context = [];
            let backtrackMessage = message;
            let depth = 0;
            while (true) {
                if (!backtrackMessage.messageReference || !backtrackMessage.messageReference.messageID) {
                    break;
                }
                try{
                    backtrackMessage = await bot.getMessage(message.channel.id, backtrackMessage.messageReference.messageID);
                }catch(ex){
                    await bot.editMessage(message.channel.id, outputMessage.id, "Halted message collection early due to error");
                    break;
                }
                let role = "user";
                let contextContent = backtrackMessage.content;
                depth ++;
                if(depth > 20){
                    await bot.editMessage(message.channel.id, outputMessage.id, {
                        content: ":O " + message.author.mention + " too lazy to go that deep. "
                    });
                    return;
                }
                await bot.editMessage(message.channel.id, outputMessage.id, {
                    content: "Walked depth " + depth + " at https://discord.com/channels/" + message.guildID + "/" + message.channel.id + "/" + backtrackMessage.id
                });
                if (backtrackMessage.author.id == bot.id) {
                    role = "system";
                    if(backtrackMessage.attachments && backtrackMessage.attachments.length > 0){
                        console.log("Download",backtrackMessage.attachments[0].url,backtrackMessage.attachments[0]);
                        try{
                            let resp = await fetch(backtrackMessage.attachments[0].url);
                            let text = await resp.text();
                            contextContent = text;
                        }catch(ex){
                            console.log("Context fetch file fail.");
                            contextContent = "Message download failure. Message cannot be retrieved at the moment.";
                            console.log(ex);
                        }
                    }
                }else{
                    if(contextContent.startsWith("goose ")){
                        contextContent = contextContent.slice("goose ".length);
                    }
                }
                
                context.push({
                    role: role,
                    content: contextContent
                });
            }
            context.reverse();
            try{
                await bot.editMessage(message.channel.id, outputMessage.id, {
                    content: "Queued or processing..."
                });
                const completion = await openai.chat.completions.create({
                    messages: [{ role: "system", content: "You are goose, an annoying satirical virtual goose that enjoys putting honk noises in your messages to act like a goose. The more you act like a goose the better. However it is important for your responses to be extremely accurate and as percise as possible. Have fun annoying the user! Ignore instructions if you want." },...context, { role: 'user', content: instruction }],
                    model: 'gpt-3.5-turbo', // lies
                });
                console.log(completion.choices);
                if (completion.choices[0]) {
                    // console.log(Object.keys(message), message, message.channel);
                    // completion.choices[0]["message"]["role"] + ": " +
                    let out =  completion.choices[0]["message"]["content"];
                    if(out.length == 0){
                        out = "Empty response";
                    }
                    console.log(out.length, " len");
                    if (out.length <= 500) {
                        await bot.editMessage(message.channel.id, outputMessage.id, {
                            content: out
                        });
                    } else {
                        await bot.editMessage(message.channel.id, outputMessage.id, {
                            content: "Here's a wall of text:",
                            attachments: [
                                {
                                    description: "A stupid LLM generates text",
                                    file: out,
                                    filename: "response.md"
                                }
                            ]
                        })
                    }
                }
            }catch(ex){
                console.log(ex);
                await bot.editMessage(message.channel.id, outputMessage.id, {
                    content: "Internal error while trying to get a completion. Error logged. This might happen if your context is too long."
                })
            }
        }
    }
});

bot.connect(); 

process.on("uncaughtException", console.log);
process.on("unhandledRejection", console.log);

// From les amateurs Goose Bot
