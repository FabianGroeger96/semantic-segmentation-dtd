apt-get update

# install git
apt-get -y install git

# install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
apt-get -y install git-lfs

# pip
pip install --upgrade pip
pip install -r requirements.txt
