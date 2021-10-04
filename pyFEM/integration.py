# Gauss-Legendre Quadrature
class integration:
    def __init__(self, num):
        self.num = 3  # number of sample points (0-4)
        self.Xi = []  # roots of the nth Legendre polynomial
        self.w = []  # Quadrature weights

        if num == 0:
            self.Xi.append(0.)
            self.w.append(2.)
        elif num == 1:
            self.Xi.append(-0.5773503)
            self.Xi.append(0.5773503)

            self.w.append(1.)
            self.w.append(1.)
        elif num == 2:
            self.Xi.append(-0.7745967)
            self.Xi.append(0.)
            self.Xi.append(0.7745967)

            self.w.append(0.5555556)
            self.w.append(0.8888889)
            self.w.append(0.5555556)
        elif num == 3:
            self.Xi.append(-0.8611363)
            self.Xi.append(-0.3399810)
            self.Xi.append(0.3399810)
            self.Xi.append(0.8611363)

            self.w.append(0.3478548)
            self.w.append(0.6521452)
            self.w.append(0.6521452)
            self.w.append(0.3478548)
        elif num == 4:
            self.Xi.append(-0.9061798)
            self.Xi.append(-0.5384693)
            self.Xi.append(0.)
            self.Xi.append(0.5384693)
            self.Xi.append(0.9061798)

            self.w.append(0.2369269)
            self.w.append(0.4786287)
            self.w.append(0.5688889)
            self.w.append(0.4786287)
            self.w.append(0.2369269)
        else:
            assert True, "Only 0-4 sample points can be allowed."