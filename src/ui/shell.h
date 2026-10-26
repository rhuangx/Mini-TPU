/// The command shell: the terminal loads programs.

#pragma once

#include <string>
#include <vector>

#include "core/tpu.h"

class Shell {
public:
    explicit Shell(Tpu& tpu);

    // Runs one terminal line, load FILE or quit, and returns the text to show; errors come back as "error: ..." text.
    std::string execute(const std::string& line);

    bool quit_requested() const;

private:
    Tpu& tpu_;
    bool quit_ = false;

    std::string load(const std::vector<std::string>& words);
};
