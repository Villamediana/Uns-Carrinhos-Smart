# Uns Carrinhos Smart

Carrinhos que aprendem a dirigir sozinhos.

Cada carrinho tem uma rede neural pequena e cinco sensores que medem a distância até a borda da pista. No começo eles batem em tudo. A cada geração os que foram mais longe viram pais da próxima, com um pouco de mutação, e depois de algumas voltas eles fazem a pista inteira.

Na tela dá pra ver a pista, a rede do melhor carrinho acendendo em tempo real e o gráfico de evolução por geração. Velocidade, tamanho da população, taxa de mutação e pista dá pra mudar enquanto roda.

## Rodar

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python app.py
```

Abre em http://localhost:5055.

O melhor cérebro fica salvo em `data/brain.json`, então ao reabrir eles continuam de onde pararam.

## Treinar sem tela

`trainer.js` roda a mesma simulação em Node, sem desenhar nada. Ele só treina quando ninguém está com a página aberta e devolve o campeão pro servidor.

```bash
node trainer.js
```

Pra deixar rodando direto, um serviço de usuário do systemd resolve:

```ini
# ~/.config/systemd/user/carrinhos-trainer.service
[Unit]
Description=Uns Carrinhos Smart · treino em segundo plano

[Service]
WorkingDirectory=%h/Uns-Carrinhos-Smart
ExecStart=/usr/bin/node trainer.js
Restart=always

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now carrinhos-trainer
```

## Como é por dentro

```
app.py              servidor Flask: página, /api/brain e /api/training
static/js/nn.js     rede neural feedforward (tanh), mutação de pesos e de neurônios
static/js/car.js    física do carrinho e sensores
static/js/tracks.js pistas
static/js/main.js   simulação, evolução e desenho
trainer.js          treino headless em Node
```
