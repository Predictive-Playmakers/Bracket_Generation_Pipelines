#!/bin/bash

# Check if the user provided the required arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 FUNCTION_NAME CODE_ENTRYPOINT"
  exit 1
fi

# Assign variables based on command-line arguments
FUNCTION_NAME="$1"
CODE_ENTRYPOINT="$2"

# Deploy the function
gcloud functions deploy "$FUNCTION_NAME" \
  --gen2 \
  --region="us-east4" \
  --runtime="python312" \
  --entry-point="$CODE_ENTRYPOINT" \
  --env-vars-file env.yaml \
  --trigger-http \
  --allow-unauthenticated \
  --memory="1GB" \
  --timeout="300s"

# Confirm completion
if [ $? -eq 0 ]; then
    echo "Function '$FUNCTION_NAME' deployed successfully."
else
    echo "Deployment of function '$FUNCTION_NAME' failed."
fi
