<h1>
Space Invaders Q-Learning
</h1>

## ⚙️ Setup

Para fazer o setup do projeto, basta rodar o script de instalação, ele vai instalar as dependências do python e importar a ROM do Space Invaders:
```bash
./scripts/install.sh
```

## 🧠 Treinamento

Para rodar o treinamento, basta executar o arquivo main:
```bash
python ./main.py
```

Caso deseje renderizar o agente treinando, você pode passar uma flag para o treinamento:
```bash
python ./main.py -r
```

## ▶️ Replay

Ao longo do treinamento, o agente vai salvar alguns checkpoints, você pode fazer o replay de  qualquer um deles com o comando usando a flag `-c` e o caminho para o checkpoint:
```bash
python ./main.py -c ./models/2021-08-20-20-20-20/checkpoints/100.check
```

Ao fim do treinamento, o modelo completo vai ser salvo, vocẽ pode fazer um replay dele com a flag `-m` e o caminhos para o modelo:
```bash
python ./main.py -m ./models/2021-08-20-20-20-20/agent.model
```

<br/>
---
<h4>Victor Costa - BCC IV</h4>