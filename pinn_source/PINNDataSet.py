class PINNDataSet:

    def __init__(self):

        self._num_points = {
            'train': {
                'bc': {},
                'pde': {},
                'data': {},
                'reg': {},
            },
            'test': {
                'bc': {},
                'pde': {},
                'data': {},
                'reg': {},
            }
        }

        self._points = {
            'train': {
                'bc': {},
                'pde': {},
                'data': {},
                'reg': {},
            },
            'test': {
                'bc': {},
                'pde': {},
                'data': {},
                'reg': {},
            }
        }

        self._batch_size_fraction = 1.0

    def get_labels(self, category, learning_type):
        return self._num_points[learning_type][category].keys()
    def set_batch_size_fraction(self, batch_size):
        self._batch_size_fraction = batch_size
    def set_points(self, num, name, category, learning_type):
        self._num_points[learning_type][category][name] = num

    def get_data(self, name, categroy, learning_type):
        data = self._points[learning_type][categroy][name]
        num = self._num_points[learning_type][categroy][name]
        num_batched = int(num * self._batch_size_fraction)
        return data, num, num_batched