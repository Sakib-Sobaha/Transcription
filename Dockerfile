FROM python:3.11
LABEL mantainer="Synesis AI"

RUN pip install --upgrade pip
RUN apt update 
RUN apt install -y wget


WORKDIR /asr-pipeline


# Update package lists and install required packages
RUN apt-get update && \
    apt-get install -y wget tar build-essential libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Download and install OpenSSL
RUN wget -O - https://www.openssl.org/source/openssl-1.1.1u.tar.gz | tar zxf - && \
    cd openssl-1.1.1u && \
    ./config --prefix=/usr/local && \
    make -j $(nproc) && \
    make install_sw install_ssldirs

# Clean up temporary files
RUN rm -rf /tmp/openssl-1.1.1u

# Configure ldconfig and set SSL_CERT_DIR environment variable
RUN ldconfig -v && \
    echo 'export SSL_CERT_DIR=/etc/ssl/certs' >> /root/.bashrc

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "-W ignore", "api.py"]