import cohere
import webbrowser


video_urls = {
    0: "https://www.youtube.com/watch?v=WDAnJpOuhI8&list=PLEoM_i-3sen_w5IYh0d5xtnpLHJeeO8l5&index=1&t=2834s&ab_channel=LesleytalksCS%2CGraphicsandFilm",
    1: "https://www.youtube.com/watch?v=63lbnckNkZ4&list=PLEoM_i-3sen_w5IYh0d5xtnpLHJeeO8l5&index=2&ab_channel=LesleytalksCS%2CGraphicsandFilm",
    2: "https://www.youtube.com/watch?v=Qid5OMJJ3x4&list=PLEoM_i-3sen_w5IYh0d5xtnpLHJeeO8l5&index=3&ab_channel=LesleytalksCS%2CGraphicsandFilm",
}

def process(response, video_urls):
    citation_docs = response.citations
    # print(citation_docs)
    indexes_array = [entry['document_ids'][0] for entry in citation_docs]
    print("INDEX OF VIDEO")
    numbers_after_underscore = [int(entry.split('_')[1].split(':')[0]) if '_' in entry else None for entry in indexes_array]
    print(numbers_after_underscore)
    # webbrowser.open(youtube_video_url)

def openVid():
    co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel')
    response = co.chat(
    message= "what is a deadlock",
    documents= [
        {
        "title": "1",
        "snippet": "The host of CS350 gives an introduction to the course and outlines the new lecture times. They go over their background, explaining they are a lecturer at Waterloo and their research area is in computer graphics, specifically in stereo 3D and pre-visualization and post-production algorithms. The lecturer also mentions their previous experience in low-level systems as a compiler and driver developer for the Linux kernel. The course will be conducted entirely online, and the instructor goes over the course website and Piazza, where all communications will take place. They also explain the grading scheme and the structure of the course, and that the main assignments will be done in C, not C++. The instructor also mentions that there is a limit to the libraries that can be used, as the assignments are compiled for a simulated MIPS architecture. They also provide a summary of the resources available to students on the course website, including assignment guides, sample problems, and lecture content. \n\nWould you like me to help you with anything else regarding this summary?"
    },
    {
        "title": "2",
        "snippet": "In this episode, the instructor provides updates on the course website, recommends students use Piazza for communication, and offers advice on troubleshooting issues with VS Code and OS161. The instructor then shares insights on computing history and recommends visiting the Computer History Museum in California. The episode concludes with a discussion on operating systems, threading, and concurrency, and how these concepts play a vital role in computer programming. \n\nWould you like me to help you with anything else?"
    },
    {
        "title": "3",
        "snippet": "In this episode, we discuss how to have multiple threads sharing a single CPU while making the user believe that all those threads are actually executing at the same time. We talk about the details of how this is done and how we can debug OS 161. We also talk about solid-state memory, the size of words on this machine, its optimized fortran compiler, and its flexible linker and loader. We also talk about the differences between a thread and a process."
    },
    {
        "title": "4",
        "snippet": "In this episode, the concept of context switching is discussed in the context of allowing multiple threads to share a single CPU while giving the illusion to the user that all of the threads are making progress. The four reasons for a context switch are covered, which include thread yield, thread exit, blocking, and preemption. The history of operating systems is then discussed, focusing on MCP (Master Control Program) from Burroughs Corporation, which was written in the high-level language Espol and supported virtual memory, multi-programming, and multi-processing. The importance of MCP is highlighted due to it being one of the first operating systems to be written in a high-level language and the first to commercially implement virtual memory. The episode ends with a demonstration of a context switch in code, as well as a discussion of the function call thread switch in other operating systems. \n\nWould you like me to help you with anything else regarding this episode?"
    },
    {
        "title": "5",
        "snippet": "In this episode, the consequences of using multi-threads in an application are discussed. The main focus is on locks and lock use, which are important topics in the synchronization module. The importance of using a real thread library, like pthreads, for assignment one, is also highlighted. Additionally, it is recommended to avoid using pthread join, and instead, use synchronization primitives to mimic the behavior of join. Some tips for completing the assignment are provided, such as using a testing harness to generate meaningful feedback. The importance of finding the right balance between the number of threads and the amount of work assigned to each thread is emphasized. The history of time-sharing systems is discussed, focusing on CTSS (Compatible Time-Sharing System), which was the first operating system to demonstrate the feasibility of time-sharing. Some of the features of CTSS are mentioned, such as text formatting utilities and the first user messaging program, which was the precursor to email. The predecessor of Unix, Multics, is also derived from CTSS. The problem of shared memory in multi-threaded applications is discussed, using an example of a project that processes 32,000 film trailers to extract color palettes."
    },
    {
        "title": "6",
        "snippet": "In this episode, Leslie continues the discussion on synchronization by looking at locks, semaphores, and condition variables. A lock is a synchronization mechanism used to enforce mutual exclusion when multiple threads are in a critical section of code. To prevent race conditions, it is important to ensure that only one thread can hold the lock at a time. Leslie then discusses load-link and store-conditional instructions, which are hardware-based operations that can be used to implement locks efficiently. \n\nThe load-link instruction reads the value of a memory location and stores it in a temporary variable, while the store-conditional instruction updates the memory location only if it has not been changed since the load-link operation. These instructions can be used to create a loop that repeatedly tries to acquire the lock until it becomes available. \n\nThe OS of the day is DTSS, which was released in 1964 and ran on a GE 200 series machine. DTSS was designed to demonstrate the feasibility of time-sharing on a large scale, and by 1968, it was being used by 80% of the students at Dartmouth College. DTSS introduced several innovations, such as the first IDE and the ability to automatically compile, execute, and test code."
    },
    {
        "title": "7",
        "snippet": "In this episode, we continue the discussion on semaphores by looking at some interesting use cases for them. We also take a closer look at the implementation of semaphores, specifically the order of the w-chan lock and spin lock release functions. Additionally, we talk about condition variables and their role in the synchronization process. The assigned reading for this episode is posted on the screen, and students are reminded to avoid using pthread join for assignment one. The focus os for the week is Multix, which was developed in 1969 and is the predecessor of Unix. It offered single-level storage, dynamic linking, and the ability to reconfigure hardware while the operating system was running. Multix had different levels of security, a hierarchical file system with symbolic links, and per-process stacks within the kernel. The next episode will cover the implementation of condition variables."
    },
    {
        "title": "8",
        "snippet": "In this episode, Leslie wraps up the discussion on synchronization by talking about other sources where race conditions can appear. She highlights how race conditions can appear through no fault of your own with modern architectures and compilers. Then she talks about the problem of deadlock which is one of the side effects of synchronization. Afterwards, she discusses OS 360, its popularity, and why it eventually became obsolete. She also talks about the history of operating systems, including OS 360, Unix, and Multix, and how they have influenced each other. \n\nThen, there is a Q&A section where the first question asks whether spam clicking can block other interrupts from doing work. Leslie explains that it is possible to do so, but it is not an easy task and nowadays, CPUs are designed to prevent such behavior. The second question asks why spam clicking sometimes can wake up a frozen computer, to which Leslie explains that sleep and hibernation modes have improved over the years, but they still are not perfect and depend on the implementation and architecture. \n\nFinally, Leslie goes over some of the key points of the episode, including the use of condition variables to solve the producer-consumer problem, and how they can be used to write more efficient code."
    },
    {
        "title": "9",
        "snippet": "In this episode, the professor introduces the concept of processes, going over their definition and components. She explains the difference between a process and a program, stating that a process is an execution environment for a program. She mentions the structure of a process and goes over the role of the kernel in creating and managing processes. She also talks about the address space of a process, explaining that each program needs an address space to run. The episode ends with a summary of the components of a process and a look at the process structure in code. \n\nNote: The professor mentions that there was a power outage due to heavy rain, which caused some of the computers and systems in the CS servers to shut down. She also mentions that there may be a deadline extension due to this incident. \n\nThe unix operating system and its influence on current operating systems is also discussed briefly."
    },
    {
        "title": "10",
        "snippet": "In this episode, we discuss process management calls within the kernel. We talk about system calls - these are the interface between the user and the kernel. We then discuss unprivileged and privileged code and how the CPU understands these concepts through layers of privilege. We also touch on the concept of adjusting the voltage of a CPU, which is related to the concept of overclocking and underclocking. \n\nPick (also known as Pic) was an operating system developed in 1965 by TRW Inc. It was one of Unix's major competitors in the 1980s. Pick was a monolithic system, and it is still in use today in the form of a database environment that can run on top of operating systems like Windows. \n\nSome amusing videos can be found on YouTube of conferences like COM DEV where people present and demonstrate Pick."
    },
    {
        "title": "11",
        "snippet": "In this episode of CS350 Online, Leslie Eistedt discusses the importance of starting Assignment 2 early, as the synchronization and deadlock issues can be difficult to debug. She then presents an overview of Alto Executive, an operating system developed at Xerox Park in 1973 that introduced many features that are still used today, such as time sharing, concurrency, and Ethernet support. Alto Executive also had a mouse-operated desktop environment and broke the isolation between kernel and user programs to optimize user applications. The episode concludes with a discussion of the implementation of locks and condition variables in Assignment 2, which is necessary before attempting the other parts of the assignment."
    },
    {
        "title": "12",
        "snippet": "In this episode, the concept of virtual memory is introduced. The address space, address translation, and segmentation are discussed. It is recommended that students start working on assignment 2 as early as possible because it takes a long time to debug. The most common errors from assignment 2 are also covered. The episode then moves on to discuss BSD, which was released in 1978 and was the first Unix-like operating system to have an IP stack. Virtual memory, which is an abstraction of physical memory, is then discussed in detail. The address space, address translation, and segmentation are explained, and it is explained why stack and heap need to be placed at opposite ends of the address space. Common misconceptions about address space layout are dispelled. The episode concludes with a brief history of how programs used to use physical memory addresses and how processes would search for a contiguous block of memory that fit their address space. \n\nWould you like me to help you with anything else regarding this episode?"
    },
    {
        "title": "13",
        "snippet": "In this episode, we continue our discussion on virtual memory, specifically on a type of virtual memory called paging. We discuss the basics of paging and how it is implemented in operating systems. We also talk about the common os161 errors that occur when running tests and the importance of reading the common errors post on Piazza. We then discuss the operating system of the day, OpenVMS, which has been around since 1977. We talk about its development history, its features, and why it is still used today. We also discuss the downside of virtual memory, the need for translations, and how this is done via the Memory Management Unit (MMU). Additionally, we cover dynamic relocation, segmentation, and the differences and similarities between the two."
    },
    {
        "title": "14",
        "snippet": "In this episode, we continue the discussion on virtual memory, focusing on techniques to improve the performance of multi-level paging. We discuss the history of Sun OS (also known as Solaris), highlighting its advanced networking and security features. We explain why Sun Systems failed despite their popularity and early entry into the 64-bit market. We then review the concept of multi-level paging, explaining its advantages and limitations. Finally, we work through a sample problem related to multi-level paging. \n\nNote: There is no transcript for this episode as it contains only a verbal explanation and no text. \n\nThe paging section in the textbook and the common OS 161 errors post can help with any questions. If you encounter any common errors, you can fix them right away instead of waiting for someone to respond. \n\nFeel free to ask any questions or request clarifications in the comments section, and we will get back to you as soon as possible. \n\nThank you for listening, and we hope you found this episode helpful!"
    },
    {
        "title": "15",
        "snippet": "In this episode, Leslie continues the discussion on virtual memory, focusing on on-demand paging. She explains the concept of translation look-aside buffer (TLB) and how it is used to reduce the time needed for translation. Leslie then talks about the differences between hardware-managed and software-managed TLBs, and the advantages and disadvantages of each approach. She also discusses the format of the MIPS TLB and the process of TLB eviction. \n\nThe episode then moves on to the operating system of the day, QNX, and its history. Leslie explains how QNX is a microkernel operating system that originated from the University of Waterloo. She also talks about the advantages of QNX, such as its real-time capabilities and its use in embedded systems, as well as its use in navigation systems and airplanes. \n\nThe episode ends with a reminder of the assignment due dates and a brief overview of the next episode. \n\nNote: This transcript is automatically generated and may contain errors. Please do not use it for quoting or referencing. \n\nFor an accurate transcript, please see the audio version of the episode."
    },
    {
        "title": "16",
        "snippet": "In this episode, the focus is on scheduling, with particular emphasis on single-core scheduling. The episode begins with a reminder to read the assigned readings, which include the MLFQ paper on the Lottery Scheduler and the CP paper on the Log Structured File System. Then, the history and features of IBM's Disc Operating System (DOS) are discussed, including its 8086 or 8088 processor, 4.77 MHz speed, 16 or 64 kilobytes of RAM, and FAT file system. Les then explains the origins of DOS, detailing how IBM approached Microsoft to develop an operating system, which they had originally intended to develop in-house. The various versions of DOS are mentioned, including MS-DOS, 86-DOS, DR-DOS, and DoubleDOS. The limitations of DOS are highlighted, including its lack of multitasking, GUI, mouse control, and security. The episode concludes with a discussion of job scheduling in the 1950s, including job arrival time, response time, turnaround time, and the assumption that jobs have known run times. The goal is to minimize either the average response time or the average turnaround time by selecting the best job to run next from the pool of ready jobs."
    },
    {
        "title": "17",
        "snippet": "In this episode of cs350 online, the instructor apologizes for audio issues and explains the issue with their microphone. The instructor then gives an overview of the history of OS2 and some of its features. OS2 was a collaboration between Microsoft and IBM, and was released in 1987. It featured several improvements over its predecessor, DOS, such as a protected mode, virtualization, and a GUI. However, it was not popular amongst consumers, and lost to Windows in terms of usability and price. The instructor then moves on to talk about input and output devices, and how the computer communicates with various hardware components, such as keyboards, mice, printers, and disk drives. In Linux and Unix-based systems, devices are treated like files, which allows for simple and straightforward interaction with hardware. The instructor notes that the clock is one of the most important devices in a computer, as it is responsible for implementing preemptive concurrency. \n\nWould you like me to help you summarize any other portion of this text?"
    },
    {
        "title": "18",
        "snippet": "In this episode, the focus is on disk drives and their drivers. The operating system of the day is Minix, which was created in 1987 by Andrew S. Tanenbaum. Minix was designed to be a miniature version of Unix, which is a monolithic kernel that includes all drivers and features. The original kernel of Minix was only 12,000 lines of code, which included the kernel, file system, and memory management. The concept of self-healing was introduced in Minix, where device drivers could be restarted without affecting the running process. This is a feature that is now found in most operating systems. The inspiration for Linux came from Minix, and Linus Torvalds used Minix to create the Linux kernel. \n\nThe concepts of device drivers and registers were also discussed. Devices are typically external devices that are attached to a computer, but there are also internal devices such as the clock. Every device has a set of device drivers and registers that can be used to interact with the device. Device drivers are used to control the device and registers are on-board physical devices that can be used to learn about the device or tell the device to do something."
    },
    {
        "title": "19",
        "snippet": "In this episode of CS350 Online, the instructor covers Assignment 3, which focuses on virtual memory management in OS161. Virtual memory allows programs to run in different memory locations, creating a separation between user programs and the operating system. However, there are issues with the current implementation of virtual memory in OS161, such as when the translation look-aside buffer (TLB) is full, causing a kernel panic. The assignment requires students to address this issue by implementing a least-recently-used (LRU) cache eviction strategy to handle TLB misses and prevent kernel panics when the TLB is full. Additionally, students need to ensure that the user program's code segment is read-only, as intended. The instructor provides guidance on the assignment, emphasizing the importance of completing previous assignments, especially those related to locks and condition variables, as they are fundamental for process management in an operating system. The assignment is worth 20% of the course grade, and the instructor recommends prioritizing it over other assignments to ensure sufficient time for completion. \n\nWould you like me to help you with anything else regarding this episode?"
    },
    {
        "title": "20",
        "snippet": "In this episode of CS350 Online, Leslie covers file systems. Specifically, the lecture covers the log-structured file system, various versions of Windows, and the differences between the consumer and commercial versions of Windows. The episode ends with a teaser for the next episode on the comparison of Unix and Windows. \n\nIs there anything in particular you'd like to hear more about? Feel free to provide feedback!"
    },
    {
        "title": "21",
        "snippet": "In this episode, the host discusses virtual file systems and physical file systems. She begins by apologizing for the delay in the recording due to a power outage and recommends that everyone use a UPS for their computers. The host then introduces the topic of the day, which is the Linux kernel, and provides a brief history of its development. She also mentions the different components of the Linux operating system and how it has evolved over time. The host then discusses the different layers of abstraction in the Linux kernel and how they interact with each other. She also talks about the different distributions of Linux and her personal preference for KDE. The host then moves on to the virtual file system, which is how Windows and Unix-based systems handle multiple file systems, and how they present them to the logical file system as one thing. The host also explains the physical file system and how it is divided into three levels: the logical file system, the virtual file system, and the physical file system. She also talks about the differences between Windows and Unix-based systems in how they handle virtual file systems. The host concludes by encouraging listeners to explore different desktop environments and customize their Linux experience to suit their preferences."
    },
    {
        "title": "22",
        "snippet": "In this episode, we discuss file systems in detail. Specifically, we cover Android and iOS, two of the most popular operating systems today. We go over their similarities and differences, as well as their histories. We also talk about the process of keeping apps running in the background on your phone and the differences between Android and iOS in this regard. There is also a quick reminder at the end of the episode about an upcoming assignment due date. \n\nThe reading assignment for Episode 22 is File Systems: Part II, found on Piazza. The assignment is due on July 30th. \n\nThe Os of the Day is iOS, which is a branch of the Mac OS X operating system. \n\nThe reference for this episode is: File Systems: The Missing Manual, by Brian K. Jones, Avery Pennarun, and Michael M. Swift. \n\nNote: The podcast transcript may contain inaccuracies. Please refer to the episode audio for the exact conversation."
    },
    {
        "title": "23",
        "snippet": "In this episode, the instructor provides a brief overview of virtual machines, including what they are and why they are used. The instructor then moves on to explain the operating system Temple OS, created by Terry Davis. Davis was inspired by manic episodes to create an operating system that acted as a temple for God. The operating system is peculiar in that it has a single address space, no network, and no concept of user-privilege. The instructor notes that although the operating system is simple, it has some unique and interesting features. The instructor then moves on to explain what a virtual machine is, how it works, and why it is used. The instructor also explains the difference between a virtual machine and a hypervisor. \nFinally, the instructor encourages students to look at the source code for Temple OS, as it is a very interesting case study in operating system design. The instructor also warns students against looking at offensive YouTube videos posted by the Temple OS creator. \nThe instructor then reminds students to complete the reading assignment and learn quizzes due the next day, and to start preparing for the exam which begins on August 7th. \nThe instructor also announces that the final episode of the course will be on networking and the operating system."
    },
    {
        "title": "24",
        "snippet": "In this episode of cs350, the professor discusses networking, specifically the five-layer model of networking. The five layers include the application layer, the transport layer, the network layer, the data link layer, and the physical layer. The application layer is the layer that users interact with, such as web browsers, Discord, and YouTube. Protocols for different applications live in the application layer, such as HTTP, FTP, and SMTP. The transport layer is what allows communication between applications, using sockets for network development. The network layer, link layer, and physical layer are layers that most application developers do not interact with; they are responsible for lower-level network communication. The professor also mentions the operating system's role in networking and how applications in the application layer interact with the kernel and user programs. \n\nWould you like me to help you summarize any other portions of this podcast?"
    },
    {
        "title": "25",
        "snippet": "In this episode, the instructor gives a final wrap-up of the course. The instructor reminds students that the Page Table Bonus is due tomorrow and that the quizzes are also due tomorrow. The instructor then gives details about the final assessment: it will be another quiz on Learn with somewhere between 60 and 70 questions to answer in three or four hours. The instructor advises not to worry about the number of decimal places in precision because there is an epsilon built in to give full points if the range is considered. The instructor then gives a brief history of Temple OS and explains why it is technically interesting. The instructor notes its limited features but mentions it has modern features such as support for multi-core and preemptive concurrency. The instructor then talks about the design considerations of an operating system and notes the importance of the user in the design. The instructor mentions the importance of the hardware in the design and notes the difficulty of making a portable operating system that supports multiple architectures. \n\nThe instructor concludes the episode by noting the importance of maintaining an operating system and how it can be more difficult for portable operating systems to be maintained over a long period of time."
    }
    ]
    ,
        prompt_truncation= "AUTO"
    )
    print(response)

    citation_docs = response.citations

    # print("The response is ")

    # print(citation_docs)
    # indexes_array = [entry['document_ids'][0] for entry in citation_docs]
    # val_array = [int(entry[-1]) for entry in indexes_array]
    # print(val_array)

    process(response, video_urls)

openVid()
