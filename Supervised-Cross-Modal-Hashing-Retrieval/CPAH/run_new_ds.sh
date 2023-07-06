source /home/george/Code/venvs/venv/bin/activate
cd /home/george/Code/dadh/

if [ $# -ne 3 ]; then
    echo "Illegal number of parameters, 3 arguments (device number, process name, dataset) required"
else

  python main.py train2 --flag $3 --proc $2 --bit 16 --device $1
  python main.py train2 --flag $3 --proc $2 --bit 32 --device $1
  python main.py train2 --flag $3 --proc $2 --bit 64 --device $1
  python main.py train2 --flag $3 --proc $2 --bit 128 --device $1

fi