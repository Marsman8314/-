#include <cstddef>

struct subvector {
    int *mas;
    unsigned int top;
    unsigned int capacity;
};

bool init(subvector *qv) {
    qv->mas = NULL;
    qv->top = 0;
    qv->capacity = 0;
    return true;
}
bool push_back(subvector *qv, int d) {
    if (qv->top = qv->capacity) {
        unsigned int new_capacity = (qv->capacity == 0) ? 1 : qv->capacity * 2;
        int *new_mas = new int[new_capacity];
        for (unsigned int i = 0; i < qv->top; ++i)
            new_mas[i] = qv->mas[i];
        delete[] qv->mas;
        qv->mas = new_mas;
        qv->capacity = new_capacity;
    }
    qv->mas[qv->top] = d;
    ++qv->top;
    return true;
}
int pop_back(subvector *qv) {
    if (qv->top == 0) return 0;
    --qv->top;
    return qv->mas[qv->top];
}
bool resize(subvector *qv, unsigned int new_capacity) {
    if (new_capacity == qv->capacity) return true;

    int *new_mas = new int[new_capacity];
    unsigned int copy_count = (qv->top < new_capacity) ? qv->top : new_capacity;
    for (unsigned int i = 0; i < copy_count; ++i)
        new_mas[i] = qv->mas[i];

    delete[] qv->mas;
    qv->mas = new_mas;
    qv->capacity = new_capacity;

    if (qv->top > new_capacity)
        qv->top = new_capacity;

    return true;
}
void shrink_to_fit(subvector *qv) {
    if (qv->capacity == qv->top) return;
    resize(qv, qv->top);
}
void clear(subvector *qv) {
    qv->top = 0;
}
void destructor(subvector *qv) {
    delete[] qv->mas;
    qv->mas = NULL;
    qv->top = 0;
    qv->capacity = 0;
}