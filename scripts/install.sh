clear
echo '📦 instalando as dependências ...\n'

pip install tensorflow gym gym-retro
echo '✅ dependências instaladas com sucesso!\n'

echo '🎮 importando ROM ...'
python -m retro.import ./rom
echo '✅ ROM importada com sucesso!'