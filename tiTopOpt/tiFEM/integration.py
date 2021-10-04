from init import *


# Gauss-Legendre Quadrature
class integration:
    def __init__(self, num):
        self.num = 3  # number of sample points (0-4)
        self.Xi = ti.field(ti.f64, shape=(num+1))  # roots of the nth Legendre polynomial
        self.w = ti.field(ti.f64, shape=(num+1)) # Quadrature weights

        if num == 0:
            self.Xi[0] = 0.
            self.w[0] = 2.
        elif num == 1:
            self.Xi[0] = -0.5773503
            self.Xi[1] = 0.5773503

            self.w[0] = 1.
            self.w[1] = 1.
        elif num == 2:
            self.Xi[0] = -0.7745967
            self.Xi[1] = 0.
            self.Xi[2] = 0.7745967

            self.w[0] = 0.5555556
            self.w[1] = 0.8888889
            self.w[2] = 0.5555556
        elif num == 3:
            self.Xi[0] = -0.8611363
            self.Xi[1] = -0.3399810
            self.Xi[2] = 0.3399810
            self.Xi[3] = 0.8611363

            self.w[0] = 0.3478548
            self.w[1] = 0.6521452
            self.w[2] = 0.6521452
            self.w[3] = 0.3478548
        elif num == 4:
            self.Xi[0] = -0.9061798
            self.Xi[1] = -0.5384693
            self.Xi[2] = 0.
            self.Xi[3] = 0.5384693
            self.Xi[4] = 0.9061798

            self.w[0] = 0.2369269
            self.w[1] = 0.4786287
            self.w[2] = 0.5688889
            self.w[3] = 0.4786287
            self.w[4] = 0.2369269
        else:
            assert True, "Only 0-4 sample points can be allowed."


if __name__ == '__main__':
    ti.init()
    a = integration(4)
    print(a.Xi)
    print(a.w)