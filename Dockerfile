FROM python:3.11-slim
LABEL mantainer="Synesis AI"

RUN pip install --upgrade pip
RUN apt update 
RUN apt-get install -y wget


WORKDIR /asr-pipeline

COPY requirements.txt .

COPY . .

RUN pip install -r requirements.txt

COPY . .

RUN wget -O - https://www.openssl.org/source/openssl-1.1.1u.tar.gz | tar zxf -
RUN cd openssl-1.1.1u
RUN ./config --prefix=/usr/local
RUN make -j $(nproc)
RUN sudo make install_sw install_ssldirs
RUN sudo ldconfig -v
RUN export SSL_CERT_DIR=/etc/ssl/certs
RUN cd ..

CMD ["python", "-W ignore", "api.py"]