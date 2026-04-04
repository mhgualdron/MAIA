#!/bin/bash
set -e

echo "Building Lambda Deployment Package..."
cd aws-async-poller

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
    else
        echo "Error: pip or pip3 not found. Please install Python pip."
        exit 1
    fi
else
    PIP_CMD="pip"
fi

# Clean previous build
rm -rf package/ deployment_package.zip

# Install psycopg2-binary for Amazon Linux compatibility and requests
$PIP_CMD install --platform manylinux2014_x86_64 --target=package --implementation cp --python-version 3.11 --only-binary=:all: --upgrade -r requirements.txt

# Copy the lambda function code
cp lambda_function.py package/

# Zip the package
cd package
zip -r ../deployment_package.zip .
cd ..

echo "Lambda package created successfully at aws-async-poller/deployment_package.zip"
