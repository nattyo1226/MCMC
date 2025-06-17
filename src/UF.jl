module UF

export UnionFind,get_root, unite!

struct UnionFind
    parent::Array{Int64, 1}

    function UnionFind(N::Int64)
        new([i for i in 1:N])
    end
end

function get_root(
    union_find::UnionFind,
    index::Int64,
)::Int64
    index_parent = union_find.parent[index]
    if (index_parent == index)
        return index
    else
        return get_root(union_find, index_parent)
    end
end

# function has_same_root(
#     union_find::UnionFind,
#     index_1::Int64,
#     index_2::Int64,
# )
#     root_1 = get_root(union_find, index_1)
#     root_2 = get_root(union_find, index_2)

#     return root_1 == root_2
# end

function unite!(
    union_find::UnionFind,
    index_1::Int64,
    index_2::Int64,
)
    root_1 = get_root(union_find, index_1)
    root_2 = get_root(union_find, index_2)

    if (root_1 != root_2)
        union_find.parent[max(root_1, root_2)] = min(root_1, root_2)
    end
end

end