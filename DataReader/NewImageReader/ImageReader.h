//
// <copyright company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "ReaderInterfaces.h"
#include "ImageTransformers.h"
#include "FrameModePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class ImageReader : public Reader
    {
    public:
        ImageReader(const ConfigParameters& parameters,
            size_t elementSize);

        std::vector<InputDescriptionPtr> GetInputs() override;
        void StartEpoch(const EpochConfiguration& config) override;
        Minibatch ReadMinibatch() override;

    private:
        void InitFromConfig(const ConfigParameters& config);

        TransformerPtr m_transformer;
        FrameModePackerPtr m_packer;
        unsigned int m_seed;

        bool m_imgListRand;
        size_t m_elementSize;
    };

}}}