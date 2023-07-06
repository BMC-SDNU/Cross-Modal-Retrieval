source /home/george/Code/venvs/venv/bin/activate
cd /home/george/Code/dadh/

if [ $# -ne 2 ]; then
    echo "Illegal number of parameters, 2 arguments (device number, process name) required"
else

  python main.py train2 --flag ucm --proc $2 --bit 16 --device $1
  python main.py train2 --flag ucm --proc $2 --bit 32 --device $1
  python main.py train2 --flag ucm --proc $2 --bit 64 --device $1
  python main.py train2 --flag ucm --proc $2 --bit 128 --device $1


  python main.py train2 --flag rsicd --proc $2 --bit 16 --device $1
  python main.py train2 --flag rsicd --proc $2 --bit 32 --device $1
  python main.py train2 --flag rsicd --proc $2 --bit 64 --device $1
  python main.py train2 --flag rsicd --proc $2 --bit 128 --device $1

fi