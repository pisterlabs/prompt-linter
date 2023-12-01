#!/usr/bin/env python3

import aws_cdk as cdk
import cdk_nag

from guidance_for_environmental_impact_factor_mapping_on_aws.eifm_stack import EifmStack

# Create CDK app
app = cdk.App()

# Create single stack within that app
EifmStack(app, "EifmStack", description="Guidance for Environmental Impact Factor Mapping on AWS (SO9244)")

# Run cdk-nag on the app
cdk.Aspects.of(app).add(cdk_nag.AwsSolutionsChecks())

app.synth()
