: '
NOTES:
Build the docker image by running this function from the docker directory to set up the environment to run the project ;)
This assumes you have docker installed
'

image="boolean_reservoir"
version="1.0"
container="boolean"
build_context=.

build_base_image()
{
    docker build -t $image:$version $build_context
}

mkdir -p ../data
mkdir -p ../out
build_base_image
