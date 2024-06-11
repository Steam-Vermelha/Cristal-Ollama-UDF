# Python-services
## Configurando ambiente com Poetry
-> [Install Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

-> [Install Pycharm](https://www.jetbrains.com/pt-br/pycharm/)

### Configure o poetry no pycharm:
https://www.jetbrains.com/help/pycharm/poetry.html#poetry-env

para testar a instalação você pode executar:
````shell
poetry
````

### Instale as dependências com o poetry
```shell
poetry install
```

### Adicionando dependências com Poetry:
````
poetry add <dependence>
````

### Atualizando as dependências de acordo com o arquivo pyproject.toml
````shell
poetry update
````

## Install language model OLLAMA:
Instale o [Ollama](https://ollama.com/) para usar a aplicação 

### Puxe o modelo de linguagem:
```shell
ollama pull llama3
```

## Configurando Pytorch:

Instale [Microsoft Visual C++ Redistributable](aka.ms/vs/16/release/vc_redist.x64.exe) para configuração de DLL.

O PyTorch é uma biblioteca extensa que utiliza muitas bibliotecas de baixo nível escritas em C e C++ para desempenho e 
eficiência. Essas bibliotecas dependem das bibliotecas de tempo de execução C++ fornecidas pelo Visual C++ Redistributable.

# Instalação do Conda e Jupyter Notebook

Este guia irá orientá-lo na instalação do Conda e do Jupyter Notebook, duas ferramentas essenciais para o desenvolvimento em Python e análise de dados.

## 1. Instalando o Conda

Conda é um sistema de gerenciamento de pacotes e ambientes que simplifica a instalação de pacotes e a gestão de ambientes virtuais. Recomendamos o uso do Anaconda ou Miniconda para instalar o Conda.

### 1.1. Instalando o Miniconda

Miniconda é uma versão minimalista do Anaconda, fornecendo apenas o Conda e seus pacotes dependentes. Siga as instruções abaixo para instalar o Miniconda:

1. **Baixar o Miniconda:**

   - Acesse a [página de downloads do Miniconda](https://docs.conda.io/en/latest/miniconda.html) e baixe a versão apropriada para o seu sistema operacional (Windows, macOS ou Linux).

2. **Instalar o Miniconda:**

   - **Windows:**
     - Execute o instalador `.exe` baixado e siga as instruções na tela.

   - **macOS e Linux:**
     - Abra o terminal e navegue até o diretório onde o instalador foi baixado.
     - Execute o seguinte comando (substitua `Miniconda3-latest-Linux-x86_64.sh` pelo nome do arquivo baixado):
       ```sh
       bash Miniconda3-latest-Linux-x86_64.sh
       ```
     - Siga as instruções na tela.

### 1.2. Verificar a Instalação

Após a instalação, abra um terminal (ou Anaconda Prompt no Windows) e verifique se o Conda foi instalado corretamente:
```sh
conda --version
```

## 2. Instalando o Jupyter Notebook

execute o código a seguir para instalação do Jupyter Notebook
```sh
pip install notebook
```

execute o Jupyter notebook com o notebook criado para gerar embeddings:

```sh
jupyter notebook Embeddings_Generate.ipynb
```
