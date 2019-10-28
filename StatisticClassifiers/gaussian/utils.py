def classes(y):
  classes = []
  for c in y:
    if c not in classes:
      classes.append(c)

  return classes
