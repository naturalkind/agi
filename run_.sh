#!/bin/bash

# Активируем виртуальное окружение Python
source venv/bin/activate

# Запускаем скрипты Python
python serv_2.py &

# Создаем новую вкладку в окне терминала
gnome-terminal --tab -e "bash -c 'source venv/bin/activate && python serv_t.py'"
# Функция для безопасного завершения скриптов при нажатии Ctrl+C
trap 'kill $(jobs -p)' SIGINT

# Ожидаем завершения выполнения всех скриптов
wait
