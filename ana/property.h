#pragma once

#define ATTRIBUTE(type, name) \
public: \
  const type& get_##name() const { return name; } \
  void set_##name(const type& v) { name = v; } \
protected: \
  type name

