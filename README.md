# Ciclo de Vida de Modelos de Machine Learning com MLflow

![Static Badge](https://img.shields.io/badge/Status-Finalizado-green)

## Descrição

Este projeto implementa um ciclo de vida completo de modelos de Machine Learning utilizando MLflow. Um script Python permite treinar modelos XGBoost com hiperparâmetros configuráveis, registrando experimentos, métricas, artefatos e modelos no MLflow Tracking. A infraestrutura suporta armazenamento local e em banco de dados SQLite. O projeto utiliza o MLflow Tracking para rastreabilidade, o MLflow Models para servir os modelos e o MLflow Registry para gerenciamento de modelos (versionamento, estágios de produção/teste). Modelos em produção são empacotados em imagens Docker utilizando o MLflow.

## Tecnologias Utilizadas

- Python
- MLflow
- Scikit-learn
- XGBoost
- SQLite
- Docker

## Funcionalidades Detalhadas

### Script de Treinamento:

- Aceita hiperparâmetros do XGBoost como argumentos de linha de comando, permitindo flexibilidade na experimentação.
- Registra os seguintes dados no MLflow Tracking para cada execução:
    - Hiperparâmetros utilizados
    - Métricas de desempenho (ex: RMSE, Rˆ2, etc.)
    - Artefatos do modelo (ex: gráficos de importância de features)
    - Modelo treinado

### Armazenamento de Dados:

- Suporte a duas opções de backend para o MLflow:
    - **Local:** Armazena os dados do MLflow no sistema de arquivos local.
    - **SQLite:** Utiliza um banco de dados SQLite para armazenar os dados do MLflow, permitindo persistência e colaboração em equipe.

### Gerenciamento de Modelos:

- **MLflow Models:** Modelos treinados são salvos utilizando o formato MLflow Model, simplificando o processo de deploy.
- **MLflow Registry:**
    - Registro de modelos treinados com versionamento automático.
    - Categorização de modelos em diferentes estágios: `staging`, `produção` ou `teste`.
    - Transição de modelos entre os estágios para facilitar o processo de deploy.

### Deploy de Modelos:

- Modelos em produção são empacotados em imagens Docker utilizando o `mlflow models build-docker`.
- A imagem Docker resultante contém o modelo, as dependências necessárias e um servidor web para servir o modelo.
