FROM breakstring/gpt-sovits:latest

COPY . /opt/python311

ENV PATH="/opt/python311/bin:${PATH}"

RUN /opt/python311/bin/python3.11 -m pip install --break-system-packages --no-cache-dir "swe-rex==1.4.0" \
    && /opt/python311/bin/swerex-remote --help >/dev/null
