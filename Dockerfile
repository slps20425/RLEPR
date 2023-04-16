# Use the official Python 3.6 image as the base image
FROM python:3.6

# Install moreutils for the 'ts' command
RUN apt-get update && apt-get install -y moreutils

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Download the TA-Lib source code
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

# Extract the archive
RUN tar -xf ta-lib-0.4.0-src.tar.gz

# Change to the extracted directory
WORKDIR ta-lib

# Build and install the library
RUN ./configure --prefix=/usr \
    && make \
    && make install

# Clean up
WORKDIR /app
RUN rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set the timezone
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update package list and install vim
RUN apt-get update && \
    apt-get install -y vim
    
# Copy the entire folder structure into the container
COPY . .

# Expose port 8050 for Dash app
EXPOSE 8050

# Run the entrypoint.sh script when the container launches
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]

