from time import perf_counter


class Timer:
    def __init__(self, message='', show=True):
        self.message = message
        self.elapsed = 0
        self.show = show

    def __enter__(self):
        if self.show and self.message:
            print(f'{self.message} ... ')

        self.start = perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.show:
            print(f'elapsed time: {perf_counter() - self.start} s')
