
"""
    perm_to_lehm_idx(P::Vector{<:Integer}) -> Int

Convert a permutation `P` of `1:n` into its 1-based lexicographic index, computed with
the Lehmer code. The `n!` permutations are numbered `1, …, n!` in lexicographic order.
The ordinal patterns in this package are indexed this way.

This version allocates a work vector on every call. Inside loops, use
[`perm_to_lehm_idx!`](@ref), which reuses a preallocated one.

# Arguments
- `P`: a permutation of `1:n`, for example the result of `sortperm`.

# Returns
- The index of `P`, an integer in `1:factorial(n)`.

# Examples
```julia
StatsOrdinalPatterns.perm_to_lehm_idx([1, 2, 3])  # 1
StatsOrdinalPatterns.perm_to_lehm_idx([1, 3, 2])  # 2
StatsOrdinalPatterns.perm_to_lehm_idx([3, 2, 1])  # 6
```
"""
function perm_to_lehm_idx(
  P::Vector{<:Integer}
)::Int

  # Vector to keep track of used elements
  used = zeros(Int, length(P))
  # Length of the permutation
  n = length(P)
  # Initialize index
  index = 0

  # Iterate through the permutation
  for i in 1:n
    current_element = P[i]
    coefficient = 0

    # Count how many smaller numbers have not been used
    for j in 1:(current_element-1)
      if used[j] == 0
        coefficient += 1
      end
    end

    # Add weighted coefficient to index (factorial number system)
    rank = n - i
    index += coefficient * factorial(rank)

    # Mark current element as used
    used[current_element] = 1
  end

  return index + 1  # 1-based indexing
end


"""
    perm_to_lehm_idx!(P::Vector{<:Integer}, used::Vector{<:Integer}) -> Int

Version of [`perm_to_lehm_idx`](@ref) without allocations: convert a permutation `P` of
`1:n` into its 1-based lexicographic index, using the preallocated work vector `used`.

`used` is modified: on return, every entry is `1`. Reset it with `fill!(used, 0)`
before the next call. `P` is not modified.

# Arguments
- `P`: a permutation of `1:n`, for example the result of `sortperm`.
- `used`: work vector of length `n` whose entries are all zero on input.

# Returns
- The index of `P`, an integer in `1:factorial(n)`.

# Examples
```julia
used = zeros(Int, 3)
StatsOrdinalPatterns.perm_to_lehm_idx!([3, 1, 2], used)  # 5
fill!(used, 0)  # required before the next call
```
"""
function perm_to_lehm_idx!(
  P::Vector{<:Integer},
  used::Vector{<:Integer}
)::Int

  # Length of the permutation
  n = length(P)
  index = 0

  # Iterate through the permutation
  for i in 1:n
    current_element = P[i]
    coefficient = 0

    # Count how many smaller numbers have not been used
    for j in 1:(current_element-1)
      if used[j] == 0
        coefficient += 1
      end
    end

    # Add weighted coefficient to index (factorial number system)
    rank = n - i
    index += coefficient * factorial(rank)

    # Mark current element as used
    used[current_element] = 1
  end

  return index + 1  # 1-based indexing
end


