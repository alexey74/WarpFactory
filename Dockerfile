
FROM pkienzle/opencl_docker:latest AS ocl
# FROM quay.io/jupyter/minimal-notebook as jupyter
FROM gnuoctave/octave:9.2.0

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

COPY --from=ocl /etc/OpenCL/ /etc/OpenCL/
COPY --from=ocl /opt/intel /opt/intel
COPY --from=ocl /opt/amdgpu /opt/amdgpu
COPY --from=ocl /opt/amdgpu-pro /opt/amdgpu-pro
COPY --from=ocl /etc/ld.so.conf.d/*amdgpu*.conf /etc/ld.so.conf.d/
COPY --from=ocl /usr/bin/clinfo /usr/bin/clinfo

WORKDIR /app

RUN ldconfig \
  && git clone https://github.com/alexey74/octave-ocl \
  && tar -zcvf ocl.tgz octave-ocl \
  && octave-cli --norc --eval 'pkg install ocl.tgz'

# see https://abdallahshamy.wordpress.com/2021/06/24/gsoc-2021-how-to-setup-the-octave-kernel-for-jupyter/
RUN mkdir -p /root/.jupyter/ && \
  pip install --no-cache-dir --break-system-packages \
    octave_kernel jupyter_kernel_gateway jupyter && \
    ln -s /usr/bin/octave /usr/bin/octave-cli-jupyter

COPY octave_kernel_config.py /root/.jupyter/
COPY .gnuplot .octaverc /root/

ENV PATH="/usr/local/bin:/opt/conda/bin:${PATH}"

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

COPY . .

ENTRYPOINT ["/usr/bin/tini", "--"]
