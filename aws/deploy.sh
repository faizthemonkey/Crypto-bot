#!/bin/bash
# AWS ECS Deployment Script for Futures Trading Bot
# Prerequisites: AWS CLI configured, Docker installed

set -e

# Configuration - Update these values
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="YOUR_ACCOUNT_ID"
ECR_REPO="futures-trading-bot"
ECS_CLUSTER="trading-bot-cluster"
ECS_SERVICE="trading-bot-service"
IMAGE_TAG="latest"

echo "🚀 Starting deployment to AWS ECS..."

# Step 1: Login to ECR
echo "📦 Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Step 2: Build Docker image
echo "🔨 Building Docker image..."
docker build -t $ECR_REPO:$IMAGE_TAG ..

# Step 3: Tag image for ECR
echo "🏷️ Tagging image..."
docker tag $ECR_REPO:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

# Step 4: Push to ECR
echo "⬆️ Pushing to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

# Step 5: Update ECS service
echo "🔄 Updating ECS service..."
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --force-new-deployment --region $AWS_REGION

echo "✅ Deployment initiated! Check AWS Console for status."
echo "🔗 Your app will be available at your ALB/domain once deployment completes."
