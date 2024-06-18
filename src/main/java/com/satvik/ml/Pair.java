package com.satvik.ml;

import lombok.Getter;

@Getter
public class Pair<A, B> {
    private final A a;
    private final B b;
    private Pair(A a, B b){
        this.a = a;
        this.b = b;
    }

    public static <X, Y> Pair<X, Y> of(X a, Y b){
        return new Pair<>(a, b);
    }
}
