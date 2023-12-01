#!/usr/bin/env python3

import aws_cdk as cdk
from aws_cdk import Aspects
from cdk_nag import (
    AwsSolutionsChecks,
    NagSuppressions
)

from guidance_for_buy_it_now_on_third_party_website_on_aws.guidance_for_buy_it_now_on_third_party_website_on_aws_stack import GuidanceForBuyItNowOnThirdPartyWebsiteOnAwsStack
from guidance_for_buy_it_now_on_third_party_website_on_aws.guidance_mock_stack import MockStack


app = cdk.App()
description = "Guidance for Buy-it-Now on third party websites on AWS (SO9191)"
thirdparty_description = "Thirdparty Stack - Guidance for Buy-it-Now on third party websites on AWS (SO9191)"
buy_it_now_stack = GuidanceForBuyItNowOnThirdPartyWebsiteOnAwsStack(app, 
                        "guidance-for-buy-it-now-on-third-party-website-on-aws",
                        description=description)
third_party_stack = MockStack(app, "Thirdparty-MockStack", description=thirdparty_description)
Aspects.of(app).add(AwsSolutionsChecks(verbose=True))
NagSuppressions.add_stack_suppressions(
    stack=third_party_stack, 
    suppressions=[
        #{"id":"AwsSolutions-APIG1", "reason":"Sample Code"},
        #{"id":"AwsSolutions-APIG2", "reason":"Sample Code"},
        #{"id":"AwsSolutions-APIG4", "reason":"Sample Code"},
        #{"id":"AwsSolutions-APIG6", "reason":"Sample Code"},
        {"id":"AwsSolutions-COG4", "reason":"Using Lambda Authorizer and not Cognito user pool authorizer"},
        #{"id":"AwsSolutions-IAM4", "reason":"Sample Code"},
        #{"id":"AwsSolutions-IAM5", "reason":"Sample Code"},

        # Suppress warnings
        {"id":"AwsSolutions-APIG3", "reason":"Not using WAF in this guidance but recommended for real-world use"},
        #{"id":"AwsSolutions-DDB3", "reason":"Sample Code"},
    ]
)
NagSuppressions.add_stack_suppressions(
    stack=buy_it_now_stack, 
    suppressions=[
        #{"id":"AwsSolutions-APIG1", "reason":"Sample Code"},
        #{"id":"AwsSolutions-APIG2", "reason":"Sample Code"},
        #{"id":"AwsSolutions-APIG4", "reason":"Sample Code"},
        #{"id":"AwsSolutions-APIG6", "reason":"Sample Code"},
        {"id":"AwsSolutions-COG4", "reason":"Using Lambda Authorizer and not Cognito user pool authorizer"},
        #{"id":"AwsSolutions-IAM4", "reason":"Sample Code"},
        #{"id":"AwsSolutions-IAM5", "reason":"Sample Code"},
        #{"id":"AwsSolutions-SNS2", "reason":"Sample Code"},
        #{"id":"AwsSolutions-SNS3", "reason":"Sample Code"},
        #{"id":"AwsSolutions-SF2", "reason":"Sample Code"},

        # Suppress warnings
        {"id":"AwsSolutions-APIG3", "reason":"Not using WAF in this guidance but recommended for real-world use"},
        #{"id":"AwsSolutions-DDB3", "reason":"Sample Code"},
    ]
)
app.synth()
