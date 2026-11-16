#!/usr/bin/env python3
import aws_cdk as cdk

from stack import BankAgentStack

app = cdk.App()
BankAgentStack(app, "BankFinancialsAgent")
app.synth()
