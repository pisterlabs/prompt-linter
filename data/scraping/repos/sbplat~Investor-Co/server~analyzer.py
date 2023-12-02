import cohere
from cohere.classify import Example

cohere_client = cohere.Client('ojsimMVIln0d6Tsd0yAprJzX47uOi8agV4L4G4Hl')

def summarize_text(text):
    response = cohere_client.generate(
        model='xlarge',
        prompt=f"""\
The following are passages and short summaries that are two sentences.
Since officially changing its name to Meta last October, the news for CEO Mark Zuckerberg and the company has been almost all bad. Apple’s iOS privacy update made it more difficult for the company to target ads and the increased popularity of social media rival TikTok has drawn users and advertisers away from the app. Meanwhile, an economic slowdown has caused many companies to pull back on their online marketing spending. In July, Meta said it was expecting a second straight period of declining sales as it reported second-quarter earnings that missed on the top and bottom lines.
TLDR: Apple, Tik Tok, and other competitors has ve drawn users away from the app. With other social media apps gaining popularity, Meta is falling more compared to the rest of the industry. 
--
FedEx Corp. saw its biggest stock drop since at least 1980 after withdrawing its earnings forecast on worsening business conditions, a potentially worrying sign for the global economy. The package-delivery giant flagged weakness in Asia and challenges in Europe as it pulled its prior outlook and reported preliminary results for the latest quarter that fell well short of Wall Street’s expectations. The conditions could deteriorate further in the current period, FedEx said. The company will take immediate steps to cut costs, including parking some aircraft, cutting workers’ hours and closing more than 90 of its roughly 2,200 FedEx Office locations. Put simply, it was an “ugly quarter,” according to Robert W. Baird & Co. analyst Garrett Holland. “Global freight demand has significantly deteriorated.” FedEx shares tumbled 22 per cent Friday morning. While U.S. economic data has been mixed, with employment and manufacturing holding up, companies across industries are starting to paint a grimmer picture of the economy. Conditions in Asia and Europe also appear to be weighing on the U.S., where consumers are shifting spending into travel and concerts and away from online shopping.
TLDR: FedEx, a global transport company, experienced their biggest drop in stock since the late 2000's. With customers switching focus to travel and in-person events following the pandemic, online shopping is the last thing on their minds.
--
{text}
TLDR:""",
        max_tokens=60,
        temperature=0,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0,
        stop_sequences=["--"],
        return_likelihoods='NONE'
    )
    return response.generations[0].text.rstrip("\n--").strip()

def classify_text(summary):
    response = cohere_client.classify(
        model='large',
        inputs=[summary],
        examples=[
            Example("Tesla, a car company, has produced 10,000 Model Y cars in Texas. The company is celebrating the milestone and is looking forward to producing more.", "Positive"),
            Example("A Tesla owner in Canada claimed he is locked out of his vehicle after the battery died, with the electric automaker telling him the replacement would cost $26,000 (Rs 20 lakh).", "Negative"), 
            Example("Tesla, a car company, is being sued by two employees who were laid off. The company is being accused of violating federal laws by requiring workers to sign separation agreements.", "Negative"), 
            Example("The CEO of Facebook, Mark Zuckerberg, has successfully navigated the company to one of the most valuable companies in the world", "Positive"), 
            Example("Facebook is losing out on communication with businesses. With the pandemic, businesses are losing out on communication with customers.", "Negative"), 
            Example("Meta Platforms Inc. stock underperforms Friday when compared to competitors.", "Negative"), 
            Example("Meta, a social media company, has partnered with MeitY to launch a startup accelerator program. The program will provide grants of up to INR 20 Lakh each for 40 startups involved in technological innovation.", "Positive"), 
            Example("The smartphone accessories market is expected to grow by 2029. The market is expected to grow due to the increase in the number of smartphone users.", "Positive"), 
            Example("Apple Watch Series 6 is the best smartwatch on the market. It has a lot of features that make it stand out from the rest.", "Positive"), 
            Example("Microsoft is looking to build a fiber optic cable that will connect Seattle to Japan. This will allow Microsoft to have a faster connection to Japan, which will allow them to have a faster connection to the rest of the world", "Positive"), 
            Example("The U.S military has finally received the first order of Microsoft\'s HoloLens goggles. The military has been testing the goggles for a while now and has been impressed with the results.", "Positive"),
            Example("This angers some MPs who do not want to see increased funding for a company whose software was recently affected by two large-scale cyberattacks.", "Negative")
        ]
    )

    return response.classifications[0].prediction == "Positive"

if __name__ == "__main__":
    summary = summarize_text("Tesla has crossed another significant manufacturing milestone. As caught by Electrek, the automaker shared on Saturday that its Texas Gigafactory recently produced its ten thousandth Model Y SUV. The achievement could be good news for those hoping to buy a Cybertruck next year. Tesla plans to build the pickup truck primarily in Texas. The automaker initially expected to begin volume production in 2021 but then delayed the Cybertruck to 2022 and then 2023.")
    print(summary)
    print(classify_text(summary))
