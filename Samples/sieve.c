extern float sqrtf(float);
void sieve(int n, int* prime)
{
    int sqrt = sqrtf(n);
    for (int p = 2; p <= sqrt; p++) {
        if (prime[p]) {
            for (int i = p * p; i <= n; i += p)
                prime[i] = 0;
        }
    }
}
