# Audio verification with TensorFlow

Did you ever want to have your own Audio verification/authentication application?
You can now! See the live video [here](https://youtu.be/NNJXNIL3dfQ) to
learn how to do it in TensorFlow.

# Installation
Requirements
- You need to have [Docker](https://docs.docker.com/engine/installation/) installed

Steps
- Go to data/scripts/files/first and add 100-200 wav files of one person saying something
- Go to data/scripts/files/second and add 100-200 wav files of another person saying the same thing
- Go to data/scripts/files/test and add test wavs from these people (saying the same thing)

Run in root folder,
~~~~
docker-compose build && docker-compose up -d
~~~~

Login to the container,
~~~~
docker exec -it tensorflow /bin/bash -c "TERM=$TERM exec bash"
~~~~

Go to /scripts folder and run
~~~~
python tf.py
~~~~

# By SocialNerds
* [SocialNerds.gr](https://www.socialnerds.gr/)
* [YouTube](https://www.youtube.com/SocialNerdsGR)
* [Facebook](https://www.facebook.com/SocialNerdsGR)
* [Twitter](https://twitter.com/socialnerdsgr)