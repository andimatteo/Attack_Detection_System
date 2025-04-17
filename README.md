# 📡 Attack Detection System
> todo: make readme


## 📦 Installazione dipendenze (da eseguire solo la prima volta)

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.charm.sh/apt/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/charm.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" | sudo tee /etc/apt/sources.list.d/charm.list
sudo apt update && sudo apt install gum tshark wget unzip python3-pip -y
pip3 install -r requirements.txt
