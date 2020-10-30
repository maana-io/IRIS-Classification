#! /usr/bin/env sh

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters. Must provide docker-registry, version-number"
    exit
fi

registry=$1
version=$2

docker build -t classification .
docker tag classification ${registry}/classification:${version}
docker push ${registry}/classification:${version}

kubectl delete service classification
kubectl delete deployment classification

cat classification_deployment.yaml | sed -e "s/{{docker-registry}}/${registry}/g" -e "s/{{version}}/${version}/g"> finalized_classification_deployment.yaml

#kubectl apply -f classification_pvc.yaml
kubectl apply -f finalized_classification_deployment.yaml
kubectl apply -f classification_service.yaml
