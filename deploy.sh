#! /usr/bin/env sh

docker build -t classification .
docker tag classification mkdrservices.azurecr.io/classification:v3.0.36
docker push mkdrservices.azurecr.io/classification:v3.0.36

kubectl delete service classification
kubectl delete deployment classification

#kubectl apply -f classification_pvc.yaml
kubectl apply -f classification_deployment.yaml
kubectl apply -f classification_service.yaml
