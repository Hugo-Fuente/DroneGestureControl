# Visão Computacional Aplicada ao Controle Inteligente de Drones

![Status](https://img.shields.io/badge/status-concluído-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)

[cite_start]Repositório do Trabalho de Conclusão de Curso de Graduação em [...Insira o nome do seu curso aqui...], desenvolvido por Hugo Gomes de la Fuente.

## 📝 Descrição

Este projeto apresenta um sistema de controle para o drone DJI Tello EDU, utilizando uma Interface Humano-Máquina (IHM) baseada em visão computacional. O objetivo foi criar uma alternativa de pilotagem intuitiva e acessível, substituindo controles tradicionais por interações com gestos manuais e modos de voo autônomos.

O sistema utiliza a biblioteca MediaPipe para o reconhecimento de gestos e a OpenCV para a detecção de faces, permitindo um controle rico e interativo da aeronave.

## ✨ Funcionalidades Principais

* **Controle por Gestos:** Pilote o drone usando diferentes posturas da mão para comandos de decolagem, pouso, movimentos direcionais (frente, trás, cima, baixo) e rotações.
* **Modo "Me Siga":** O drone detecta um rosto com sua própria câmera, o centraliza e mantém uma distância constante, seguindo o usuário de forma autônoma.
* **Modo "Scan de Rostos":** A aeronave executa uma varredura panorâmica de 360 graus, detectando e contabilizando o número de faces no ambiente.
* **Interface e Feedback em Tempo Real:** Uma janela do OpenCV exibe o status do drone (bateria, modo atual), o gesto detectado e legendas de comandos, fornecendo um feedback crucial para o operador.
* **Segurança:** Comandos críticos como decolagem e pouso exigem a manutenção do gesto por um período, evitando ativações acidentais.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Bibliotecas Principais:**
    * **OpenCV:** Para processamento de imagem, detecção de faces (Haar Cascades) e criação da interface visual.
    * **MediaPipe:** Para detecção de mãos e rastreamento dos 21 landmarks em tempo real.
    * **DJITelloPy:** Para abstrair a comunicação e o envio de comandos para o drone.
    * **NumPy:** Para manipulação de arrays e operações numéricas.
* **Hardware:**
    * Drone DJI Tello EDU 
    * Notebook com Webcam

## 🔧 Instalação e Configuração

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    ```
    * No Windows, ative com:
        ```bash
        .\venv\Scripts\activate
        ```
    * No macOS/Linux, ative com:
        ```bash
        source venv/bin/activate
        ```

3.  **Instale as dependências:**
    Crie um arquivo `requirements.txt` com o conteúdo abaixo e depois execute `pip install -r requirements.txt`.

    **requirements.txt:**
    ```
    opencv-python
    mediapipe
    djitellopy
    numpy
    ```

## 🚀 Como Executar

1.  Ligue o drone DJI Tello EDU.
2.  Conecte seu computador à rede Wi-Fi gerada pelo drone.
3.  Execute o script principal do projeto:
    ```bash
    python seu_script_principal.py
    ```
    *(**Nota:** Renomeie `seu_script_principal.py` para o nome do seu arquivo principal)*

4.  Uma janela do OpenCV será aberta, mostrando a imagem da sua webcam.
5.  Siga as instruções exibidas na tela. [cite_start]Para a decolagem inicial, posicione-se em frente à webcam e faça o gesto de **Mão Fechada**, segurando por 1.5 segundos para iniciar a contagem regressiva.

## ✋ Dicionário de Gestos

### Navegação e Modos

| Gesto | Ação | Estado Necessário |
| :--- | :--- | :--- |
| **Mão Fechada** (segurar) | Decolar (na 1ª vez) / Armar Controle de Gestos | `INICIALIZANDO` / `CONTROLE_GESTAO` |
| **Mão Aberta** (segurar) | Pousar / Voltar ao Menu | Qualquer modo ativo |
| **Indicador + Mínimo** | Ativar Modo `CONTROLE_GESTAO` | `MENU` |
| **Indicador + Médio** | Ativar Modo `MODO_COMANDO` | `MENU` |

### Controle de Voo (após "armar")

| Gesto | Ação | Comando RC |
| :--- | :--- | :--- |
| **4 Dedos** (sem polegar) | Mover para Frente | `rc_fwd_bwd = 25` |
| **"L"** (Polegar + Indicador) | Mover para Trás | `rc_fwd_bwd = -25` |
| **Indicador** | Subir | `rc_up_down = 25` |
| **"V"** (Indicador + Médio) | Descer | `rc_up_down = -25` |
| **Polegar** | Mover para Esquerda | `rc_left_right = -25` |
| **Mínimo** | Mover para Direita | `rc_left_right = 25` |
| **Polegar + Mínimo** | Rotacionar Direita | `rc_yaw = 25` |
| **Polegar + Indicador + Médio** | Rotacionar Esquerda | `rc_yaw = -25` |

### Comandos Especiais

| Gesto | Ação | Estado Necessário |
| :--- | :--- | :--- |
| **Polegar + Mínimo** | Ativar "Me Siga" | `MODO_COMANDO` |
| **Indicador + Médio + Anelar**| Ativar "Scan de Rostos" | `MODO_COMANDO` |
| **Polegar + Indicador + Mínimo**| Tirar Foto | `CONTROLE_GESTAO` (Armado) |

## 👨‍💻 Autor

* **Hugo Gomes de la Fuente**
    * [LinkedIn](URL_DO_SEU_LINKEDIN)
    * [GitHub](URL_DO_SEU_GITHUB)

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
