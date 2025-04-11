void quick(int* a, int lo, int hi){
    if (lo >= hi)
        return;
    int pivot = a[hi];
    int i = lo;
    for(int j = lo; j <= hi; j++){
        if (a[j] < pivot){
            int tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
            i++;
        }
    }
    a[hi] = a[i];
    a[i] = pivot;
    quick(a, lo, i - 1);
    quick(a, i + 1, hi);
}
