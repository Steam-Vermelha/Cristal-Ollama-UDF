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