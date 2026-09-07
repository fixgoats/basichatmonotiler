#ifndef BASICHATMONOTILE_SMALLARR_H
#define BASICHATMONOTILE_SMALLARR_H
#include <array>
#include <exception>
#include <ostream>
#include <iostream>

#ifndef NDEBUG
#define ASSERT(condition, message)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__         \
                << " line " << __LINE__ << ": " << message << std::endl;       \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)
#else
#define ASSERT(condition, message)                                             \
  do {                                                                         \
  } while (false)
#endif

template <class T, size_t Cap>
struct SmallArr : std::array<T, Cap> {
  size_t len;

  struct Iterator {
    T* m_ptr;

    Iterator& operator++() noexcept {
      this->m_ptr++;
      return *this;
    }
    Iterator& operator--() noexcept {
      this->m_ptr--;
      return *this;
    }
    Iterator operator++(int) noexcept {
      Iterator tmp = *this;
      this->m_ptr++;
      return tmp;
    }
    Iterator operator--(int) noexcept {
      Iterator tmp = *this;
      this->m_ptr--;
      return tmp;
    }
    T& operator*() noexcept { return *this->m_ptr; }
    bool operator==(const Iterator& other) const noexcept {
      return this->m_ptr == other.m_ptr;
    }
    bool operator!=(const Iterator& other) const noexcept {
      return this->m_ptr != other.m_ptr;
    }
  };
  struct ConstIterator {
    const T* m_ptr;

    ConstIterator& operator++() noexcept {
      this->m_ptr++;
      return *this;
    }
    ConstIterator& operator--() noexcept {
      this->m_ptr--;
      return *this;
    }
    ConstIterator operator++(int) noexcept {
      Iterator tmp = *this;
      this->m_ptr++;
      return tmp;
    }
    ConstIterator operator--(int) noexcept {
      Iterator tmp = *this;
      this->m_ptr--;
      return tmp;
    }
    const T& operator*() noexcept { return *this->m_ptr; }
    bool operator==(const ConstIterator& other) const noexcept {
      return this->m_ptr == other.m_ptr;
    }
    bool operator!=(const ConstIterator& other) const noexcept {
      return this->m_ptr != other.m_ptr;
    }
  };

  constexpr SmallArr() = default;

  template <class... Args>
  constexpr SmallArr(Args&&... args)
    requires(std::is_same_v<std::common_type_t<Args...>, T>)
      : std::array<T, Cap>{std::forward<Args>(args)...}, len{sizeof...(Args)} {}
  constexpr SmallArr(size_t s) : len{s}, std::array<T, Cap>{} {}

  [[nodiscard]] constexpr T operator[](auto i) const noexcept {
    ASSERT(i < len, "Attempted out of bounds access.");
    return this->data()[i];
  }

  [[nodiscard]] constexpr T& operator[](auto i) const noexcept {
    ASSERT(i < len, "Attempted out of bounds access.");
    return this->data()[i];
  }

  [[nodiscard]] T& operator[](auto i) noexcept {
    ASSERT(i < len, "Attempted out of bounds access.");
    return this->data()[i];
  }

  [[nodiscard]] constexpr T back() const noexcept {
    ASSERT(len > 0, "Attempted to access empty array.");
    return this->data()[len - 1];
  }

  [[nodiscard]] T& back() noexcept {
    ASSERT(len > 0, "Attempted to access empty array.");
    return this->data()[len - 1];
  }

  constexpr void push_back(T x) noexcept {
    ASSERT(len < Cap, "Pushing back would exceed capacity.");
    this->data()[len] = x;
    len += 1;
  }

  template <class... Args>
  constexpr void emplace_back(Args&&... args) noexcept {
    ASSERT(len < Cap, "Emplacing back would exceed capacity.");
    this->data()[len] = T{std::forward<Args>(args)...};
    len += 1;
  }

  [[nodiscard]] ConstIterator cbegin() const noexcept {
    return ConstIterator{this->data()};
  }
  [[nodiscard]] ConstIterator cend() const noexcept {
    return ConstIterator{this->data() + this->len};
  }
  [[nodiscard]] ConstIterator begin() const noexcept { return cbegin(); }
  [[nodiscard]] ConstIterator end() const noexcept { return cend(); }
  Iterator begin() noexcept { return Iterator{this->data()}; }
  Iterator end() noexcept { return Iterator{this->data() + this->len}; }
};

#undef ASSERT
#endif // BASICHATMONOTILE_SMALLARR_H
