.. _installation:

Installation
------------

BiaPy can be locally installed/run on any Linux, Windows and Mac OS platform using `Docker <docker.html>`__ or via command line (with Anaconda/Miniconda and Git). Alternatively, the BiaPy can also be used on `Google Colab <colab.html>`__.


.. _installation_command_line:

Command line installation
~~~~~~~~~~~~~~~~~~~~~~~~~

For command line usage, you only need to set-up a ``conda`` environment. For that you first need to install `Anaconda/Miniconda <https://www.anaconda.com/>`__. For a more detailed installation based on your operating system: `Windows <https://docs.anaconda.com/anaconda/install/windows/>`__, `macOS <https://docs.anaconda.com/anaconda/install/mac-os/>`__ and `Linux <https://docs.anaconda.com/anaconda/install/linux/>`__. 

Afterwards, you will need `git <https://git-scm.com/>`__ which is a free and open source distributed version control system. It will allow you to download the code easily with just a command. You can download and install it `here <https://git-scm.com/downloads>`__. For a more detailed installation based on your operating system: `Windows <https://git-scm.com/download/win>`__, `macOS <https://git-scm.com/download/mac>`__ and `Linux <https://git-scm.com/download/linux>`__. 

Once Anaconda and git are installed you need to use a terminal for the following steps. You can open a terminal in each of operating systems following these steps: 

* In **Windows** you have installed a terminal called ``Git Bash`` if you followed git installation above. Find that application in the Start menu by clicking on the Windows icon and typing ``Git Bash`` into the search bar. Click on it to open the ``Git Bash`` terminal.
* In **macOS** you already have Bash terminal installed so just open it. If you have never done it before find more info `here <https://support.apple.com/en-ie/guide/terminal/apd5265185d-f365-44cb-8b09-71a064a42125/mac>`__.
* In **Linux** you already have Bash terminal installed so just open it. If you have never done it before find more info `here <https://www.geeksforgeeks.org/how-to-open-terminal-in-linux/>`__.

Then you are prepared to download `BiaPy <https://github.com/danifranco/BiaPy>`__ repository by running this command in the terminal ::

    git clone https://github.com/danifranco/BiaPy.git

This will create a folder called ``BiaPy`` that contains all the files of the `library's official repository <https://github.com/danifranco/BiaPy>`__. Then you need to create a ``conda`` environment using the file located in `BiaPy/utils/env/environment.yml <https://github.com/danifranco/BiaPy/blob/master/utils/env/environment.yml>`__ ::
    
    conda env create -f BiaPy/utils/env/environment.yml


Docker installation
~~~~~~~~~~~~~~~~~~~

To run BiaPy using Docker you need to install it first `here <https://docs.docker.com/get-docker/>`__.

.. Firstly check that the code will be able to use a GPU by running: ::

..     docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

.. Build the container or pull ours: ::

..     # Option A)
..     docker pull danifranco/em_image_segmentation

..     # Option B)
..     cd BiaPy
..     docker build -f utils/env/Dockerfile -t em_image_segmentation .


Google Colab
~~~~~~~~~~~~

Nothing special is needed except a browser on your PC.

